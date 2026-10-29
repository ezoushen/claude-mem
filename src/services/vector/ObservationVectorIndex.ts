import type { Database } from 'bun:sqlite';
import { createHash } from 'node:crypto';
import type { Embedder } from './types.js';
import { projectScopeSql } from '../sqlite/project-read-keys.js';
import { DEFAULT_PLATFORM_SOURCE, normalizePlatformSource } from '../../shared/platform-source.js';
import { logger } from '../../utils/logger.js';

/**
 * Search by meaning for the local worker, inside claude-mem.db.
 *
 * The storage is #3694's: one row per document in a table of its own, the
 * embedding as a little-endian float32 BLOB, the model's identity on every row
 * and the parent held by a foreign key with ON DELETE CASCADE, so a deleted
 * observation takes its vector with it. Scope (project, merged_into_project,
 * platform_source) is not copied onto the vector row; it is read from the
 * observation at query time, so a project merge needs no vector rewrite.
 *
 * Two things differ from #3694, both to match what was measured (#3610):
 *
 *  - One document per observation: title, subtitle and narrative (or the
 *    legacy text when there is no narrative), joined by newlines. That is the
 *    document the evaluation embedded; Chroma's per-field split (narrative,
 *    text, each fact) was not measured, and with 1024-dimension vectors it
 *    would multiply the table several times over.
 *  - Observations only. Summaries and prompts keep keyword search.
 *
 * A scan of every row through SQLite took 5 to 8 s at 205k rows of 1024
 * floats, so search keeps the matrix in memory, refreshed from the table by
 * rowid. There each vector is 8-bit integers with one scale (a quarter of the
 * float32 size: 218 MB for 213k vectors against 870 MB), scored with ONNX
 * Runtime matrix products over float32 chunks. That ranking is approximate, so
 * the best candidates (at least 100, four times what was asked for) are read
 * back from the table as float32 and re-scored exactly. On the 205k-note store
 * that equals scoring every float32 vector (fused nDCG@10 0.707 both ways; the
 * 8-bit scores alone lost 0.004). It can differ only among vectors whose
 * cosines to the query agree within about 1e-3, hundreds of near-duplicates
 * deep, where the returned notes are practically tied with the exact ones.
 * If the ONNX op is unavailable a JavaScript loop scores the chunks instead.
 */

export const OBSERVATION_VECTOR_TABLE = 'vec_observation_docs';

export function ensureObservationVectorSchema(db: Database): void {
  db.run(`
    CREATE TABLE IF NOT EXISTS ${OBSERVATION_VECTOR_TABLE} (
      doc_id       TEXT PRIMARY KEY,
      sqlite_id    INTEGER NOT NULL,
      field_type   TEXT NOT NULL,
      fact_index   INTEGER,
      content_hash TEXT NOT NULL,
      model_id     TEXT NOT NULL,
      dims         INTEGER NOT NULL,
      embedding    BLOB NOT NULL,
      FOREIGN KEY (sqlite_id) REFERENCES observations(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS ix_${OBSERVATION_VECTOR_TABLE}_sqlite ON ${OBSERVATION_VECTOR_TABLE}(sqlite_id);
    CREATE INDEX IF NOT EXISTS ix_${OBSERVATION_VECTOR_TABLE}_model ON ${OBSERVATION_VECTOR_TABLE}(model_id);
  `);
}

export interface ObservationTextRow {
  id: number;
  title: string | null;
  subtitle: string | null;
  narrative: string | null;
  text: string | null;
}

/** The one document embedded for an observation; '' when it has none. */
export function observationDocText(row: ObservationTextRow): string {
  return [row.title, row.subtitle, row.narrative || row.text]
    .map(part => (part ?? '').trim())
    .filter(part => part.length > 0)
    .join('\n');
}

// Rows whose document would be empty are never pending: they would be
// selected again on every pass.
const HAS_TEXT_SQL = `(COALESCE(TRIM(o.title), '') <> '' OR COALESCE(TRIM(o.subtitle), '') <> ''
  OR COALESCE(TRIM(o.narrative), '') <> '' OR COALESCE(TRIM(o.text), '') <> '')`;

export interface ObservationVectorScope {
  /** Read keys, matched against project and merged_into_project. */
  projects?: string[];
  platformSource?: string;
}

export interface ObservationVectorHit {
  id: number;
  /** Cosine similarity; higher is nearer. */
  score: number;
}

/**
 * Rows widened to float32 for each matrix product. 4,096 rows took 240 ms per
 * search at 212k vectors, 16,384 took 150 ms and 65,536 took 145 ms; the buffer
 * is 64 MB at 1024 dimensions.
 */
const SCORE_CHUNK_ROWS = 16384;

/** Candidates re-scored from the table: this many, or this multiple of the hits asked for, if more. */
const RESCORE_MIN = 100;
const RESCORE_FACTOR = 4;
/** Candidate ids per rescore query, under SQLite's limit on bound parameters. */
const RESCORE_BATCH = 500;

interface MatrixCache {
  ids: Int32Array;
  /** Row-major, dims per row: each vector as round(component / scale), in -127..127. */
  matrix: Int8Array;
  /** Per row: largest absolute component / 127, so component ~ matrix * scale. 0 for a vector of zeros. */
  scales: Float32Array;
  /** Rows held in ids, matrix and scales. */
  count: number;
  /** Table rows read, malformed ones included, to compare with COUNT(*). */
  seen: number;
  maxRowid: number;
  /** sqlite_id of the row at maxRowid, to notice that row being replaced (SQLite reuses a freed highest rowid). */
  tailId: number;
}

function emptyCache(): MatrixCache {
  return { ids: new Int32Array(0), matrix: new Int8Array(0), scales: new Float32Array(0), count: 0, seen: 0, maxRowid: 0, tailId: 0 };
}

type MatmulOp = (matrix: Float32Array, rows: number, dims: number, probe: Float32Array) => Promise<Float32Array>;

export class ObservationVectorIndex {
  private cache: MatrixCache | null = null;
  private matmul: Promise<MatmulOp | null> | null = null;
  /** Reused by every search to widen 8-bit rows to float32. */
  private widened: Float32Array | null = null;
  private scoring: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly db: Database,
    private readonly embedder: Embedder,
  ) {
    ensureObservationVectorSchema(db);
  }

  get modelId(): string {
    return this.embedder.modelId;
  }

  /** Bytes the in-memory vectors occupy; 0 until a search has loaded them. */
  matrixBytes(): number {
    return this.cache ? this.cache.count * (this.embedder.dims + Float32Array.BYTES_PER_ELEMENT) : 0;
  }

  /** Whether the embedding model has loaded; an embedder without a load is always ready. */
  isModelReady(): boolean {
    return this.embedder.isReady?.() ?? true;
  }

  /** Observations without a vector from this model, newest first. */
  pendingRows(limit: number): ObservationTextRow[] {
    return this.db.prepare(`
      SELECT o.id, o.title, o.subtitle, o.narrative, o.text FROM observations o
      WHERE ${HAS_TEXT_SQL}
        AND NOT EXISTS (
          SELECT 1 FROM ${OBSERVATION_VECTOR_TABLE} v WHERE v.sqlite_id = o.id AND v.model_id = ?
        )
      ORDER BY o.id DESC
      LIMIT ?
    `).all(this.embedder.modelId, limit) as ObservationTextRow[];
  }

  pendingCount(): number {
    const row = this.db.prepare(`
      SELECT COUNT(*) AS n FROM observations o
      WHERE ${HAS_TEXT_SQL}
        AND NOT EXISTS (
          SELECT 1 FROM ${OBSERVATION_VECTOR_TABLE} v WHERE v.sqlite_id = o.id AND v.model_id = ?
        )
    `).get(this.embedder.modelId) as { n: number };
    return row.n;
  }

  indexedCount(): number {
    const row = this.db.prepare(
      `SELECT COUNT(*) AS n FROM ${OBSERVATION_VECTOR_TABLE} WHERE model_id = ?`,
    ).get(this.embedder.modelId) as { n: number };
    return row.n;
  }

  /**
   * Embed and store these rows. An observation deleted while it was being
   * embedded is skipped rather than failing the batch. Returns rows written.
   */
  async index(rows: ObservationTextRow[]): Promise<number> {
    const docs = rows
      .map(row => ({ id: row.id, text: observationDocText(row) }))
      .filter(doc => doc.text.length > 0);
    if (docs.length === 0) return 0;
    const vectors = await this.embedder.embed(docs.map(doc => doc.text), 'passage');
    const write = this.db.prepare(`
      INSERT INTO ${OBSERVATION_VECTOR_TABLE}
        (doc_id, sqlite_id, field_type, fact_index, content_hash, model_id, dims, embedding)
      SELECT ?, ?, 'doc', NULL, ?, ?, ?, ?
      WHERE EXISTS (SELECT 1 FROM observations WHERE id = ?)
      ON CONFLICT(doc_id) DO UPDATE SET
        content_hash = excluded.content_hash,
        model_id = excluded.model_id,
        dims = excluded.dims,
        embedding = excluded.embedding
    `);
    let written = 0;
    this.db.transaction(() => {
      docs.forEach((doc, i) => {
        const vector = vectors[i];
        if (vector.length !== this.embedder.dims) {
          throw new Error(`embedder returned ${vector.length} dimensions, expected ${this.embedder.dims}`);
        }
        const hash = createHash('sha256').update(doc.text).digest('hex');
        const bytes = new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
        written += write.run(`obs_${doc.id}`, doc.id, hash, this.embedder.modelId, this.embedder.dims, bytes, doc.id).changes;
      });
    })();
    return written;
  }

  /**
   * The `limit` nearest observations to `text` within `scope`, nearest first.
   * Ids only: the caller hydrates them, which applies the remaining filters.
   */
  async search(text: string, scope: ObservationVectorScope, limit: number): Promise<ObservationVectorHit[]> {
    const k = Math.floor(limit);
    if (!(k > 0)) return [];
    const [probe] = await this.embedder.embed([text], 'query');
    const cache = this.refresh();
    // Rows another search appends while this one awaits land beyond this count, in place.
    const count = cache.count;
    if (count === 0) return [];
    const scores = await this.score(cache, count, probe);
    const allowed = this.allowedIds(scope);
    const candidates = topK(scores, cache.ids, count, Math.max(RESCORE_MIN, RESCORE_FACTOR * k), allowed);
    return this.rescore(candidates, probe).slice(0, k);
  }

  /**
   * Candidates with their cosine to `probe` computed from the float32 vectors in
   * the table, nearest first. A candidate whose row is gone (the observation was
   * deleted since the matrix was read) is dropped.
   */
  private rescore(candidates: ObservationVectorHit[], probe: Float32Array): ObservationVectorHit[] {
    const dims = this.embedder.dims;
    const row32 = new Float32Array(dims); // a stored row, aligned, to read from
    const row8 = new Uint8Array(row32.buffer);
    const best = new Map<number, number>();
    const unique = [...new Set(candidates.map(hit => hit.id))];
    for (let start = 0; start < unique.length; start += RESCORE_BATCH) {
      const ids = unique.slice(start, start + RESCORE_BATCH);
      // `+model_id` keeps SQLite on the sqlite_id index: with model_id indexed too it scans the whole model.
      const rows = this.db.prepare(
        `SELECT sqlite_id, embedding FROM ${OBSERVATION_VECTOR_TABLE}
         WHERE +model_id = ? AND sqlite_id IN (${ids.map(() => '?').join(',')})`,
      ).values(this.embedder.modelId, ...ids) as [number, Uint8Array | null][];
      for (const [sqliteId, blob] of rows) {
        if (!blob || blob.byteLength !== dims * 4) continue;
        row8.set(blob);
        let score = 0;
        for (let d = 0; d < dims; d++) score += row32[d] * probe[d];
        const id = Number(sqliteId);
        if (Number.isFinite(score) && !(best.get(id)! >= score)) best.set(id, score);
      }
    }
    return [...best].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score);
  }

  /** Ids inside the scope, or null for an unscoped search. */
  private allowedIds(scope: ObservationVectorScope): Set<number> | null {
    const projects = scope.projects?.filter(Boolean) ?? [];
    if (projects.length === 0 && !scope.platformSource) return null;
    const where: string[] = [];
    const params: string[] = [];
    if (projects.length > 0) {
      const sql = projectScopeSql('o', projects, { includeMerged: true });
      where.push(sql.sql);
      params.push(...sql.params);
    }
    let join = '';
    if (scope.platformSource) {
      join = 'LEFT JOIN sdk_sessions s ON s.memory_session_id = o.memory_session_id';
      where.push(`COALESCE(NULLIF(s.platform_source, ''), '${DEFAULT_PLATFORM_SOURCE}') = ?`);
      params.push(normalizePlatformSource(scope.platformSource));
    }
    const rows = this.db.prepare(
      `SELECT o.id AS id FROM observations o ${join} WHERE ${where.join(' AND ')}`,
    ).values(...params) as [number][];
    return new Set(rows.map(row => Number(row[0])));
  }

  /**
   * Bring the in-memory matrix up to date: append rows written since the last
   * look, and reload everything when the row count says rows were removed or
   * replaced in place (a model change rewrites rows under their old rowids).
   */
  private refresh(): MatrixCache {
    const modelId = this.embedder.modelId;
    const meta = this.db.prepare(
      `SELECT COUNT(*) AS n, COALESCE(MAX(rowid), 0) AS m FROM ${OBSERVATION_VECTOR_TABLE} WHERE model_id = ?`,
    ).get(modelId) as { n: number; m: number };
    const current = this.cache;
    // The last row held may have been deleted and its rowid given to another note.
    const replaced = current !== null && current.maxRowid > 0 && this.tailIdAt(current.maxRowid) !== current.tailId;
    if (current && !replaced && meta.n === current.seen && meta.m === current.maxRowid) return current;
    let cache = current && !replaced && meta.n > current.seen && meta.m > current.maxRowid
      ? this.load(current, current.maxRowid, meta.n - current.seen)
      : null;
    if (!cache || cache.seen !== meta.n) {
      const started = performance.now();
      cache = this.load(emptyCache(), 0, meta.n);
      logger.info('VECTOR_INDEX', 'Loaded vectors for search', {
        vectors: cache.count,
        megabytes: Math.round(cache.count * (this.embedder.dims + Float32Array.BYTES_PER_ELEMENT) / 1048576),
        ms: Math.round(performance.now() - started),
      });
    }
    this.cache = cache;
    return cache;
  }

  private tailIdAt(rowid: number): number | null {
    const row = this.db.prepare(
      `SELECT sqlite_id FROM ${OBSERVATION_VECTOR_TABLE} WHERE rowid = ? AND model_id = ?`,
    ).get(rowid, this.embedder.modelId) as { sqlite_id: number } | null;
    return row ? Number(row.sqlite_id) : null;
  }

  /**
   * Read the rows after `afterRowid` into the matrix one at a time: `expected`
   * rows are counted before the read so the matrix is sized once, and no copy
   * of every embedding is held beside it (a store of 212k bge-m3 vectors is
   * 870 MB, and reading the whole result first doubled the load's peak).
   */
  private load(base: MatrixCache, afterRowid: number, expected: number): MatrixCache {
    const dims = this.embedder.dims;
    const rows = this.db.prepare(
      `SELECT rowid, sqlite_id, embedding FROM ${OBSERVATION_VECTOR_TABLE}
       WHERE model_id = ? AND rowid > ? ORDER BY rowid`,
    );
    let cache = grow(base, base.count + expected, dims);
    const row32 = new Float32Array(dims); // a stored row, aligned, to convert from
    const row8 = new Uint8Array(row32.buffer);
    let skipped = 0;
    for (const row of rows.iterate(this.embedder.modelId, afterRowid) as Iterable<{ rowid: number; sqlite_id: number; embedding: Uint8Array | null }>) {
      cache.seen++;
      if (Number(row.rowid) > cache.maxRowid) {
        cache.maxRowid = Number(row.rowid);
        cache.tailId = Number(row.sqlite_id);
      }
      const blob = row.embedding;
      if (!blob || blob.byteLength !== dims * 4) {
        skipped++;
        continue;
      }
      // Rows written by another process since the count was taken.
      if (cache.count === cache.ids.length) cache = grow(cache, cache.count + 1, dims);
      row8.set(blob);
      let largest = 0;
      for (let d = 0; d < dims; d++) largest = Math.max(largest, Math.abs(row32[d]));
      const scale = largest / 127;
      const at = cache.count * dims;
      // A vector of zeros divides 0 by 0, which an Int8Array stores as 0; one with a NaN gets a NaN scale and never ranks.
      for (let d = 0; d < dims; d++) cache.matrix[at + d] = Math.round(row32[d] / scale);
      cache.scales[cache.count] = scale;
      cache.ids[cache.count] = Number(row.sqlite_id);
      cache.count++;
    }
    if (skipped > 0) logger.warn('VECTOR_INDEX', 'Skipped malformed embeddings', { skipped, dims });
    return cache;
  }

  /** One search scores at a time: they share the widening buffer, and each awaits the matrix product between chunks. */
  private score(cache: MatrixCache, count: number, probe: Float32Array): Promise<Float32Array> {
    const run = this.scoring.then(() => this.scoreRows(cache, count, probe));
    this.scoring = run.catch(() => undefined);
    return run;
  }

  private async scoreRows(cache: MatrixCache, count: number, probe: Float32Array): Promise<Float32Array> {
    const dims = this.embedder.dims;
    const need = Math.min(SCORE_CHUNK_ROWS, count) * dims;
    if ((this.widened?.length ?? 0) < need) this.widened = new Float32Array(need);
    const widened = this.widened!;
    const scores = new Float32Array(count);
    this.matmul ??= loadOnnxMatmul();
    for (let start = 0; start < count; start += SCORE_CHUNK_ROWS) {
      const rows = Math.min(SCORE_CHUNK_ROWS, count - start);
      widened.set(cache.matrix.subarray(start * dims, (start + rows) * dims));
      const part = await this.scoreChunk(widened.subarray(0, rows * dims), rows, dims, probe);
      for (let r = 0; r < rows; r++) scores[start + r] = part[r] * cache.scales[start + r];
    }
    return scores;
  }

  private async scoreChunk(chunk: Float32Array, rows: number, dims: number, probe: Float32Array): Promise<Float32Array> {
    const op = await this.matmul;
    if (op) {
      try {
        return await op(chunk, rows, dims, probe);
      } catch (error) {
        logger.warn('VECTOR_INDEX', 'ONNX matrix product failed; scoring in JavaScript', {
          error: error instanceof Error ? error.message : String(error),
        });
        this.matmul = Promise.resolve(null);
      }
    }
    return scoreInJs(chunk, rows, dims, probe);
  }
}

/**
 * Room for `rows`, plus a tenth more: untouched capacity costs no resident
 * memory, and the next notes then land without copying the whole matrix.
 */
function grow(cache: MatrixCache, rows: number, dims: number): MatrixCache {
  if (cache.ids.length >= rows) return cache;
  const capacity = Math.max(Math.ceil(rows * 1.1), 1024);
  const ids = new Int32Array(capacity);
  ids.set(cache.ids.subarray(0, cache.count));
  const matrix = new Int8Array(capacity * dims);
  matrix.set(cache.matrix.subarray(0, cache.count * dims));
  const scales = new Float32Array(capacity);
  scales.set(cache.scales.subarray(0, cache.count));
  return { ...cache, ids, matrix, scales };
}

async function loadOnnxMatmul(): Promise<MatmulOp | null> {
  try {
    const { Tensor, matmul } = await import('@huggingface/transformers') as any;
    if (typeof matmul !== 'function') return null;
    return async (matrix, rows, dims, probe) => {
      const out = await matmul(new Tensor('float32', matrix, [rows, dims]), new Tensor('float32', probe, [dims, 1]));
      return out.data as Float32Array;
    };
  } catch {
    return null;
  }
}

export function scoreInJs(matrix: Float32Array, rows: number, dims: number, probe: Float32Array): Float32Array {
  const scores = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const base = r * dims;
    let sum = 0;
    for (let d = 0; d < dims; d++) sum += matrix[base + d] * probe[d];
    scores[r] = sum;
  }
  return scores;
}

/** The k highest scores among allowed ids, best first. */
export function topK(
  scores: Float32Array,
  ids: Int32Array,
  count: number,
  k: number,
  allowed: Set<number> | null,
): ObservationVectorHit[] {
  const best: ObservationVectorHit[] = [];
  for (let r = 0; r < count; r++) {
    const score = scores[r];
    if (!Number.isFinite(score)) continue;
    if (best.length === k && score <= best[k - 1].score) continue;
    const id = ids[r];
    if (allowed && !allowed.has(id)) continue;
    let at = best.length;
    while (at > 0 && best[at - 1].score < score) at--;
    best.splice(at, 0, { id, score });
    if (best.length > k) best.pop();
  }
  return best;
}

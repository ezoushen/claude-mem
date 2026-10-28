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
 * rowid, and scores it with one ONNX Runtime matrix product (about 0.1 s),
 * falling back to a JavaScript loop (about 1 s) if that op is unavailable.
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

interface MatrixCache {
  ids: Int32Array;
  matrix: Float32Array;
  /** Rows held in ids and matrix. */
  count: number;
  /** Table rows read, malformed ones included, to compare with COUNT(*). */
  seen: number;
  maxRowid: number;
}

function emptyCache(): MatrixCache {
  return { ids: new Int32Array(0), matrix: new Float32Array(0), count: 0, seen: 0, maxRowid: 0 };
}

type MatmulOp = (matrix: Float32Array, rows: number, dims: number, probe: Float32Array) => Promise<Float32Array>;

export class ObservationVectorIndex {
  private cache: MatrixCache | null = null;
  private matmul: Promise<MatmulOp | null> | null = null;

  constructor(
    private readonly db: Database,
    private readonly embedder: Embedder,
  ) {
    ensureObservationVectorSchema(db);
  }

  get modelId(): string {
    return this.embedder.modelId;
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
    if (cache.count === 0) return [];
    const scores = await this.score(cache, probe);
    const allowed = this.allowedIds(scope);
    return topK(scores, cache.ids, cache.count, k, allowed);
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
    if (current && meta.n === current.seen && meta.m === current.maxRowid) return current;
    let cache = current && meta.n > current.seen && meta.m > current.maxRowid
      ? this.load(current, current.maxRowid)
      : null;
    if (!cache || cache.seen !== meta.n) cache = this.load(emptyCache(), 0);
    this.cache = cache;
    return cache;
  }

  private load(base: MatrixCache, afterRowid: number): MatrixCache {
    const dims = this.embedder.dims;
    const rows = this.db.prepare(
      `SELECT rowid, sqlite_id, embedding FROM ${OBSERVATION_VECTOR_TABLE}
       WHERE model_id = ? AND rowid > ? ORDER BY rowid`,
    ).values(this.embedder.modelId, afterRowid) as [number, number, Uint8Array][];
    const cache = grow(base, base.count + rows.length, dims);
    let skipped = 0;
    for (const [rowid, sqliteId, blob] of rows) {
      cache.seen++;
      cache.maxRowid = Math.max(cache.maxRowid, Number(rowid));
      if (!blob || blob.byteLength !== dims * 4) {
        skipped++;
        continue;
      }
      new Uint8Array(cache.matrix.buffer, cache.count * dims * 4, dims * 4).set(blob);
      cache.ids[cache.count] = Number(sqliteId);
      cache.count++;
    }
    if (skipped > 0) logger.warn('VECTOR_INDEX', 'Skipped malformed embeddings', { skipped, dims });
    return cache;
  }

  private async score(cache: MatrixCache, probe: Float32Array): Promise<Float32Array> {
    const dims = this.embedder.dims;
    this.matmul ??= loadOnnxMatmul();
    const op = await this.matmul;
    if (op) {
      try {
        return await op(cache.matrix.subarray(0, cache.count * dims), cache.count, dims, probe);
      } catch (error) {
        logger.warn('VECTOR_INDEX', 'ONNX matrix product failed; scoring in JavaScript', {
          error: error instanceof Error ? error.message : String(error),
        });
        this.matmul = Promise.resolve(null);
      }
    }
    return scoreInJs(cache.matrix, cache.count, dims, probe);
  }
}

function grow(cache: MatrixCache, rows: number, dims: number): MatrixCache {
  if (cache.ids.length >= rows) return cache;
  const capacity = Math.max(rows, Math.ceil(cache.ids.length * 1.5), 1024);
  const ids = new Int32Array(capacity);
  ids.set(cache.ids.subarray(0, cache.count));
  const matrix = new Float32Array(capacity * dims);
  matrix.set(cache.matrix.subarray(0, cache.count * dims));
  return { ...cache, ids, matrix };
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

#!/usr/bin/env bun
/**
 * Import observation vectors made elsewhere (for example on a GPU by
 * embed_store_mlx.py) into claude-mem.db, so the worker finds them current and
 * embeds only the notes written since.
 *
 *   bun scripts/vector-index/import-vectors.ts <claude-mem.db> <store.f32> <store.ids.json> [--model bge-m3] [--dry]
 *
 * store.f32 holds one little-endian float32 vector per id, in the order of the
 * ids file. Rows are written as ObservationVectorIndex.index() writes them,
 * under the model identity the worker computes for --model, in short write
 * transactions so a running worker is never blocked for long. A note that no
 * longer exists, or has no text, is skipped; a vector already stored for that
 * model is left alone, so the import can be run again; one from another model
 * is replaced, as the worker replaces it.
 */
import { Database } from 'bun:sqlite';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { embeddingModelId, resolveEmbeddingModel } from '../../src/services/vector/models.js';
import {
  OBSERVATION_VECTOR_TABLE,
  ensureObservationVectorSchema,
  observationDocText,
  type ObservationTextRow,
} from '../../src/services/vector/ObservationVectorIndex.js';

export interface VectorFile {
  ids: number[];
  /** ids.length * dims floats. */
  vectors: Float32Array;
  dims: number;
  modelId: string;
}

export interface ImportResult { written: number; already: number; gone: number; empty: number }

export function importVectors(db: Database, file: VectorFile, options: { dry?: boolean; batch?: number } = {}): ImportResult {
  const { ids, vectors, dims, modelId } = file;
  if (vectors.length !== ids.length * dims) {
    throw new Error(`${vectors.length / dims} vectors for ${ids.length} ids`);
  }
  if (!options.dry) ensureObservationVectorSchema(db);
  const tableExists = db.query(`SELECT 1 FROM sqlite_master WHERE name = ?`).get(OBSERVATION_VECTOR_TABLE);
  const have = new Set(tableExists
    ? (db.query(`SELECT sqlite_id FROM ${OBSERVATION_VECTOR_TABLE} WHERE model_id = ?`).values(modelId) as [number][]).map(r => r[0])
    : []);
  const read = db.query('SELECT id, title, subtitle, narrative, text FROM observations WHERE id = ?');
  // Prepared only for a real import: in a dry run the table may not exist yet.
  const write = options.dry ? null : db.query(`INSERT INTO ${OBSERVATION_VECTOR_TABLE}
    (doc_id, sqlite_id, field_type, fact_index, content_hash, model_id, dims, embedding)
    VALUES (?, ?, 'doc', NULL, ?, ?, ?, ?)
    ON CONFLICT(doc_id) DO UPDATE SET content_hash = excluded.content_hash, model_id = excluded.model_id,
      dims = excluded.dims, embedding = excluded.embedding`);
  const result: ImportResult = { written: 0, already: 0, gone: 0, empty: 0 };
  const batch = options.batch ?? 500;
  for (let start = 0; start < ids.length; start += batch) {
    const work = () => {
      for (let i = start; i < Math.min(ids.length, start + batch); i++) {
        const id = ids[i];
        if (have.has(id)) { result.already++; continue; }
        const row = read.get(id) as ObservationTextRow | null;
        if (!row) { result.gone++; continue; }
        const text = observationDocText(row);
        if (!text) { result.empty++; continue; }
        if (!write) { result.written++; continue; }
        const bytes = new Uint8Array(vectors.buffer, vectors.byteOffset + i * dims * 4, dims * 4);
        result.written += write.run(`obs_${id}`, id, createHash('sha256').update(text).digest('hex'), modelId, dims, bytes).changes;
      }
    };
    // Take the write lock first: a read that later upgrades fails at once if the worker wrote in between.
    if (options.dry) work();
    else db.transaction(work).immediate();
  }
  return result;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const flag = (name: string) => { const i = args.indexOf(name); return i < 0 ? undefined : args.splice(i, 2)[1]; };
  const dry = args.includes('--dry');
  const modelName = flag('--model') ?? 'bge-m3';
  const [dbPath, vectorPath, idsPath] = args.filter(a => a !== '--dry');
  if (!dbPath || !vectorPath || !idsPath) {
    console.error('usage: bun scripts/vector-index/import-vectors.ts <claude-mem.db> <store.f32> <store.ids.json> [--model bge-m3] [--dry]');
    process.exit(2);
  }
  if (!existsSync(dbPath)) throw new Error(`${dbPath} does not exist`);
  const model = resolveEmbeddingModel(modelName);
  const raw = readFileSync(vectorPath);
  const vectors = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const db = new Database(dbPath);
  db.run('PRAGMA busy_timeout = 30000');
  const result = importVectors(db, {
    ids: JSON.parse(readFileSync(idsPath, 'utf8')), vectors, dims: model.dims, modelId: embeddingModelId(model),
  }, { dry });
  console.log(`${embeddingModelId(model)}: ${dry ? 'would write' : 'wrote'} ${result.written}, already stored ${result.already}, note gone ${result.gone}, no text ${result.empty}`);
}

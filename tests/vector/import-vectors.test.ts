import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { importVectors } from '../../scripts/vector-index/import-vectors.js';
import { OBSERVATION_VECTOR_TABLE } from '../../src/services/vector/ObservationVectorIndex.js';
import { vectorFixture, type VectorFixture } from './support.js';

let fixture: VectorFixture;
beforeEach(() => { fixture = vectorFixture(); });
afterEach(() => { fixture.store.close(); });

const DIMS = 4;
const vectors = (n: number) => new Float32Array(Array.from({ length: n * DIMS }, (_, i) => i / 100));
const rows = () => fixture.store.db.prepare(
  `SELECT sqlite_id, model_id, dims, embedding FROM ${OBSERVATION_VECTOR_TABLE} ORDER BY sqlite_id`,
).all() as { sqlite_id: number; model_id: string; dims: number; embedding: Uint8Array }[];

describe('importVectors', () => {
  it('stores each vector under the model id for notes that exist and have text, once', () => {
    const a = fixture.addObservation('p', 'Gateway quota');
    const b = fixture.addObservation('p', 'Widget crash');
    const blank = fixture.addObservation('p', 'to be emptied');
    fixture.store.db.prepare("UPDATE observations SET title = '  ', subtitle = NULL, narrative = NULL, text = NULL WHERE id = ?").run(blank);
    const result = importVectors(fixture.store.db, { ids: [a, 9999, blank, b], vectors: vectors(4), dims: DIMS, modelId: 'm/1' });
    expect(result).toEqual({ written: 2, already: 0, gone: 1, empty: 1 });
    const stored = rows();
    expect(stored.map(r => [r.sqlite_id, r.model_id, r.dims])).toEqual([[a, 'm/1', DIMS], [b, 'm/1', DIMS]]);
    // b was the fourth vector in the file.
    expect(Array.from(new Float32Array(stored[1].embedding.slice().buffer))).toEqual(Array.from(vectors(4).subarray(3 * DIMS)));

    expect(importVectors(fixture.store.db, { ids: [a, 9999, blank, b], vectors: vectors(4), dims: DIMS, modelId: 'm/1' }))
      .toEqual({ written: 0, already: 2, gone: 1, empty: 1 });
  });

  it('refuses vectors that do not match the ids', () => {
    expect(() => importVectors(fixture.store.db, { ids: [1, 2], vectors: vectors(3), dims: DIMS, modelId: 'm/1' }))
      .toThrow(/3 vectors for 2 ids/);
  });

  it('replaces a vector another model made, as the worker does, so switching models needs no other step', () => {
    const a = fixture.addObservation('p', 'Gateway quota');
    importVectors(fixture.store.db, { ids: [a], vectors: vectors(1), dims: DIMS, modelId: 'm/old' });
    expect(importVectors(fixture.store.db, { ids: [a], vectors: vectors(1), dims: DIMS, modelId: 'm/new' }))
      .toEqual({ written: 1, already: 0, gone: 0, empty: 0 });
    expect(rows().map(r => r.model_id)).toEqual(['m/new']);
  });

  it('can say what it would write before the table exists, and writes nothing', () => {
    const fresh = vectorFixture();
    try {
      const a = fresh.addObservation('p', 'Gateway quota');
      expect(importVectors(fresh.store.db, { ids: [a], vectors: vectors(1), dims: DIMS, modelId: 'm/1' }, { dry: true }))
        .toEqual({ written: 1, already: 0, gone: 0, empty: 0 });
      expect(fresh.store.db.prepare(`SELECT 1 FROM sqlite_master WHERE name = '${OBSERVATION_VECTOR_TABLE}'`).get()).toBeNull();
    } finally {
      fresh.store.close();
    }
  });
});

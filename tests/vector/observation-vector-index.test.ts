import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import type { SessionStore } from '../../src/services/sqlite/SessionStore.js';
import {
  OBSERVATION_VECTOR_TABLE,
  ObservationVectorIndex,
  observationDocText,
  scoreInJs,
  topK,
} from '../../src/services/vector/ObservationVectorIndex.js';
import { ObservationVectorIndexer } from '../../src/services/vector/ObservationVectorIndexer.js';
import { TopicEmbedder, vectorFixture, type VectorFixture } from './support.js';

let fixture: VectorFixture;
let store: SessionStore;
let embedder: TopicEmbedder;
let index: ObservationVectorIndex;
const addObservation = (project: string, title: string, narrative: string | null = null, subtitle: string | null = null) =>
  fixture.addObservation(project, title, narrative, subtitle);

beforeEach(() => {
  fixture = vectorFixture();
  store = fixture.store;
  embedder = new TopicEmbedder();
  index = new ObservationVectorIndex(store.db, embedder);
});

afterEach(() => {
  store.close();
});

// Stored directly, as an import would: no observation rows, so the foreign key is set aside.
function storeVectors(db: SessionStore['db'], modelId: string, dims: number, from: number, to: number, vectorFor: (id: number) => Float32Array) {
  const insert = db.prepare(`INSERT INTO ${OBSERVATION_VECTOR_TABLE}
    (doc_id, sqlite_id, field_type, fact_index, content_hash, model_id, dims, embedding) VALUES (?, ?, 'doc', NULL, 'h', ?, ?, ?)`);
  db.run('PRAGMA foreign_keys = OFF');
  db.transaction(() => {
    for (let id = from; id <= to; id++) insert.run(`obs_${id}`, id, modelId, dims, new Uint8Array(vectorFor(id).buffer));
  })();
}
const WIDGET = 2; // TOPICS index of 'widget'
const QUOTA = 1;
const axis = (dims: number, at: number) => { const v = new Float32Array(dims); v[at] = 1; return v; };

describe('observationDocText', () => {
  it('joins title, subtitle and narrative, falling back to the legacy text', () => {
    expect(observationDocText({ id: 1, title: 'T', subtitle: ' S ', narrative: 'N', text: 'X' })).toBe('T\nS\nN');
    expect(observationDocText({ id: 1, title: 'T', subtitle: null, narrative: null, text: 'X' })).toBe('T\nX');
    expect(observationDocText({ id: 1, title: ' ', subtitle: null, narrative: '', text: null })).toBe('');
  });
});

describe('ObservationVectorIndex', () => {
  it('lists unindexed observations newest first and indexes them once', async () => {
    const a = addObservation('p', 'Gateway quota', 'the gateway hit its quota');
    const b = addObservation('p', 'Widget crash');
    expect(index.pendingRows(10).map(row => row.id)).toEqual([b, a]);
    expect(index.pendingCount()).toBe(2);

    expect(await index.index(index.pendingRows(10))).toBe(2);
    expect(index.pendingCount()).toBe(0);
    expect(index.indexedCount()).toBe(2);
    expect(embedder.calls.at(-1)).toEqual({ texts: ['Widget crash', 'Gateway quota\nthe gateway hit its quota'], role: 'passage' });

    const row = store.db.prepare(`SELECT doc_id, field_type, dims, length(embedding) AS bytes FROM ${OBSERVATION_VECTOR_TABLE} WHERE sqlite_id = ?`).get(a);
    expect(row).toEqual({ doc_id: `obs_${a}`, field_type: 'doc', dims: embedder.dims, bytes: embedder.dims * 4 });
  });

  it('ranks by similarity, embeds the query as a query, and scopes by project and merge', async () => {
    const quota = addObservation('alpha', 'Gateway quota exhausted');
    const widget = addObservation('alpha', 'Widget crash');
    const other = addObservation('beta', 'Gateway quota on beta');
    await index.index(index.pendingRows(10));

    const hits = await index.search('gateway quota', {}, 2);
    expect(hits.map(hit => hit.id).sort()).toEqual([quota, other].sort());
    expect(embedder.calls.at(-1)?.role).toBe('query');

    expect((await index.search('gateway quota', { projects: ['alpha'] }, 5)).map(hit => hit.id)).toEqual([quota, widget]);

    store.db.prepare('UPDATE observations SET merged_into_project = ? WHERE id = ?').run('alpha', other);
    expect((await index.search('gateway quota', { projects: ['alpha'] }, 2)).map(hit => hit.id).sort())
      .toEqual([quota, other].sort());
  });

  it('picks up new rows and drops deleted ones between searches', async () => {
    const first = addObservation('p', 'Memory search');
    await index.index(index.pendingRows(10));
    expect((await index.search('memory', {}, 5)).map(hit => hit.id)).toEqual([first]);

    const second = addObservation('p', 'Memory router');
    await index.index(index.pendingRows(10));
    expect((await index.search('memory', {}, 5)).map(hit => hit.id).sort()).toEqual([first, second].sort());

    store.db.prepare('DELETE FROM observations WHERE id = ?').run(first);
    expect(index.indexedCount()).toBe(1);
    expect((await index.search('memory', {}, 5)).map(hit => hit.id)).toEqual([second]);
  });

  it('finds every note that arrives after the first search, however many', async () => {
    const quota = addObservation('p', 'Gateway quota exhausted');
    await index.index(index.pendingRows(10));
    expect((await index.search('quota', {}, 5)).map(hit => hit.id)).toEqual([quota]);

    // More notes than the matrix first had room for.
    const routers: number[] = [];
    for (let i = 0; i < 1100; i++) routers.push(addObservation('p', `Router note ${i}`));
    while (index.pendingCount() > 0) await index.index(index.pendingRows(250));

    const found = (await index.search('router', {}, 2000)).map(hit => hit.id);
    expect(found.slice(0, routers.length).sort((a, b) => a - b)).toEqual(routers);
    // The note that was already in the matrix keeps its vector: "Gateway quota" shares one of its
    // two topics with the query "quota", so cosine is 1 / sqrt(2).
    const [first] = await index.search('quota', {}, 1);
    expect(first.id).toBe(quota);
    expect(first.score).toBeCloseTo(Math.SQRT1_2, 3);
  });

  it('treats vectors from another model as missing', async () => {
    addObservation('p', 'Deadline');
    await index.index(index.pendingRows(10));
    const other = new ObservationVectorIndex(store.db, new TopicEmbedder('fake/other'));
    expect(other.pendingCount()).toBe(1);
    expect(await other.search('deadline', {}, 5)).toEqual([]);
    await other.index(other.pendingRows(10));
    expect(other.indexedCount()).toBe(1);
    expect(index.indexedCount()).toBe(0);
  });

  it('skips an observation deleted while it was being embedded', async () => {
    const id = addObservation('p', 'Quota');
    const rows = index.pendingRows(10);
    store.db.prepare('DELETE FROM observations WHERE id = ?').run(id);
    expect(await index.index(rows)).toBe(0);
  });
});

describe('ObservationVectorIndex over more rows than one scoring chunk', () => {
  it('answers each of several concurrent searches from the whole store', async () => {
    // The first 16,400 rows lie along "widget", the next 3,600 along "quota".
    storeVectors(store.db, embedder.modelId, embedder.dims, 1, 20_000, id => axis(embedder.dims, id <= 16_400 ? WIDGET : QUOTA));
    for (let round = 0; round < 20; round++) {
      const [widgets, quotas] = await Promise.all([index.search('widget', {}, 5), index.search('quota', {}, 5)]);
      expect(widgets.map(hit => hit.id).every(id => id <= 16_400)).toBe(true);
      expect(quotas.map(hit => hit.id).every(id => id > 16_400)).toBe(true);
      expect(widgets.every(hit => Math.abs(hit.score - 1) < 1e-3)).toBe(true);
      expect(quotas.every(hit => Math.abs(hit.score - 1) < 1e-3)).toBe(true);
    }
  });

  it('keeps a search running when notes are added while it scores', async () => {
    class WideEmbedder extends TopicEmbedder { readonly dims = 1024; }
    const wideEmbedder = new WideEmbedder();
    const wide = new ObservationVectorIndex(store.db, wideEmbedder);
    storeVectors(store.db, wideEmbedder.modelId, wideEmbedder.dims, 1, 16_380, () => axis(wideEmbedder.dims, WIDGET));
    const before = wide.search('widget', {}, 3); // loads the matrix, then scores it
    for (let tick = 0; tick < 20; tick++) await Promise.resolve(); // it is now waiting on the matrix product
    // Ten more rows take the count past a chunk boundary while that search is in flight.
    storeVectors(store.db, wideEmbedder.modelId, wideEmbedder.dims, 16_381, 16_390, () => axis(wideEmbedder.dims, WIDGET));
    const after = wide.search('widget', {}, 3);
    const [first, second] = await Promise.all([before, after]);
    expect(first).toHaveLength(3);
    expect(second).toHaveLength(3);
  });
});

describe('ObservationVectorIndex holding 8-bit vectors', () => {
  const DIMS = 32;
  // A small deterministic generator, so the vectors are the same on every run.
  const random = (seed: number) => () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296 - 0.5; };
  const unit = (next: () => number) => {
    const v = Float32Array.from({ length: DIMS }, next);
    const norm = Math.hypot(...v);
    return v.map(x => x / norm);
  };
  class Fixed extends TopicEmbedder {
    readonly dims = DIMS;
    queries = new Map<string, Float32Array>();
    async embed(texts: string[]): Promise<Float32Array[]> { return texts.map(t => this.queries.get(t)!); }
  }
  const dot = (a: Float32Array, b: Float32Array) => a.reduce((sum, x, i) => sum + x * b[i], 0);

  it('returns the exact float32 ranking and scores of the stored vectors', async () => {
    const fixed = new Fixed();
    const wide = new ObservationVectorIndex(store.db, fixed);
    const next = random(7);
    const docs = Array.from({ length: 400 }, () => unit(next));
    storeVectors(store.db, fixed.modelId, DIMS, 1, 400, id => docs[id - 1]);
    for (let q = 0; q < 25; q++) {
      const probe = unit(next);
      fixed.queries.set(`q${q}`, probe);
      const expected = docs.map((doc, i) => ({ id: i + 1, score: dot(doc, probe) })).sort((a, b) => b.score - a.score).slice(0, 5);
      const hits = await wide.search(`q${q}`, {}, 5);
      expect(hits.map(hit => hit.id)).toEqual(expected.map(hit => hit.id));
      hits.forEach((hit, i) => expect(hit.score).toBeCloseTo(expected[i].score, 6));
    }
  });

  it('finds the nearest notes among many that the 8-bit scores cannot tell apart', async () => {
    // 40 vectors whose cosine to the query rises by 1e-4 per id, from 0.3. Their largest component is
    // another one, so the 8-bit rounding error on the queried component (about 3e-3) is far more than
    // the gaps: the 8-bit scores tie or misorder them, and only the re-score settles it.
    const fixed = new Fixed();
    const wide = new ObservationVectorIndex(store.db, fixed);
    const docs = Array.from({ length: 40 }, (_, i) => {
      const along = 0.3 + 1e-4 * i;
      const v = new Float32Array(DIMS);
      v[0] = along;
      v[5] = Math.sqrt(1 - along * along);
      return v;
    });
    storeVectors(store.db, fixed.modelId, DIMS, 1, 40, id => docs[id - 1]);
    fixed.queries.set('q', axis(DIMS, 0));
    const hits = await wide.search('q', {}, 3);
    expect(hits.map(hit => hit.id)).toEqual([40, 39, 38]);
  });

  it('lists a note once however many of its vectors are stored', async () => {
    // The table allows several rows per observation (one per field); the note is one hit with its best score.
    const fixed = new Fixed();
    const wide = new ObservationVectorIndex(store.db, fixed);
    const next = random(17);
    const docs = Array.from({ length: 700 }, () => unit(next));
    storeVectors(store.db, fixed.modelId, DIMS, 1, 700, id => docs[id - 1]);
    const probe = docs[4]; // note 5 is its own nearest neighbour
    fixed.queries.set('q', probe);
    // A second, weaker row for note 5, and a second row for note 300.
    store.db.prepare(`INSERT INTO ${OBSERVATION_VECTOR_TABLE} (doc_id, sqlite_id, field_type, fact_index, content_hash, model_id, dims, embedding)
      VALUES ('obs_5_b', 5, 'doc', 1, 'h', ?, ?, ?), ('obs_300_b', 300, 'doc', 1, 'h', ?, ?, ?)`)
      .run(fixed.modelId, DIMS, new Uint8Array(docs[5].buffer), fixed.modelId, DIMS, new Uint8Array(docs[301].buffer));
    for (const k of [5, 130]) {
      const hits = await wide.search('q', {}, k);
      const ids = hits.map(hit => hit.id);
      expect(new Set(ids).size).toBe(ids.length);
      expect(ids[0]).toBe(5);
      expect(hits[0].score).toBeCloseTo(1, 6);
    }
  });

  it('finds a note that took the place of the newest one after it was deleted', async () => {
    // SQLite hands the freed rowid to the next insert, so the table keeps the same size and the same
    // highest rowid while its last row is a different note.
    const next = embedder; // TopicEmbedder: "widget" and "quota" are different axes
    storeVectors(store.db, next.modelId, next.dims, 1, 20, () => axis(next.dims, QUOTA));
    expect((await index.search('quota', {}, 3)).length).toBe(3);
    store.db.prepare(`DELETE FROM ${OBSERVATION_VECTOR_TABLE} WHERE sqlite_id = 20`).run();
    storeVectors(store.db, next.modelId, next.dims, 99, 99, () => axis(next.dims, WIDGET));
    expect((await index.search('widget', {}, 3))[0]).toMatchObject({ id: 99 });
    // And the same after more notes arrive.
    storeVectors(store.db, next.modelId, next.dims, 100, 100, () => axis(next.dims, QUOTA));
    expect((await index.search('widget', {}, 3))[0]).toMatchObject({ id: 99 });
  });

  it('holds one byte per dimension and one scale per vector in memory', async () => {
    const fixed = new Fixed();
    const wide = new ObservationVectorIndex(store.db, fixed);
    const next = random(11);
    storeVectors(store.db, fixed.modelId, DIMS, 1, 50, () => unit(next));
    fixed.queries.set('q', unit(next));
    expect(wide.matrixBytes()).toBe(0); // nothing is held until a search needs it
    await wide.search('q', {}, 3);
    expect(wide.matrixBytes()).toBe(50 * DIMS + 50 * 4);
  });

  it('ignores a stored vector of zeros instead of failing on it', async () => {
    const fixed = new Fixed();
    const wide = new ObservationVectorIndex(store.db, fixed);
    const next = random(13);
    const good = unit(next);
    storeVectors(store.db, fixed.modelId, DIMS, 1, 1, () => good);
    storeVectors(store.db, fixed.modelId, DIMS, 2, 2, () => new Float32Array(DIMS));
    fixed.queries.set('q', good);
    const hits = await wide.search('q', {}, 5);
    expect(hits[0].id).toBe(1);
    expect(hits[0].score).toBeCloseTo(1, 6);
    expect(hits.every(hit => Number.isFinite(hit.score))).toBe(true);
  });
});

describe('scoring', () => {
  it('scores in JavaScript and keeps the k best allowed rows', () => {
    const matrix = new Float32Array([1, 0, 0, 1, 0.6, 0.8]);
    const scores = scoreInJs(matrix, 3, 2, new Float32Array([1, 0]));
    expect(Array.from(scores)).toEqual([1, 0, expect.closeTo(0.6, 5)]);
    const ids = new Int32Array([10, 20, 30]);
    expect(topK(scores, ids, 3, 2, null).map(hit => hit.id)).toEqual([10, 30]);
    expect(topK(scores, ids, 3, 2, new Set([20, 30])).map(hit => hit.id)).toEqual([30, 20]);
  });
});

describe('ObservationVectorIndexer', () => {
  it('does not keep picking up a note whose text is only whitespace SQLite does not trim', async () => {
    const blank = addObservation('p', 'to be blanked');
    // SQLite's TRIM strips spaces only; a newline title passes it, but the document text is empty.
    store.db.prepare("UPDATE observations SET title = char(10), subtitle = NULL, narrative = NULL, text = NULL WHERE id = ?").run(blank);
    const indexer = new ObservationVectorIndexer(index);
    expect(await indexer.runOnce()).toBe(1);
    expect(await indexer.runOnce()).toBe(0);
    expect(index.indexedCount()).toBe(0);
  });

  it('embeds pending rows in batches and isolates a row that cannot be embedded', async () => {
    addObservation('p', 'Good one');
    const bad = addObservation('p', 'Bad one');
    addObservation('p', 'Good two');
    embedder.fail = text => text.startsWith('Bad');
    const indexer = new ObservationVectorIndexer(index, { batchSize: 1 });

    expect(await indexer.runOnce()).toBe(1);
    // The model "loaded" (this embedder has no load), so the row is blamed.
    expect(await indexer.runOnce()).toBe(1);
    expect(await indexer.runOnce()).toBe(1);
    expect(await indexer.runOnce()).toBe(0);
    expect(index.pendingRows(10).map(row => row.id)).toEqual([bad]);
    expect(index.indexedCount()).toBe(2);
  });
});

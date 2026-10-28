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

import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import type { SessionStore } from '../../src/services/sqlite/SessionStore.js';
import { SessionSearch } from '../../src/services/sqlite/SessionSearch.js';
import { SearchManager } from '../../src/services/worker/SearchManager.js';
import { FormattingService } from '../../src/services/worker/FormattingService.js';
import { TimelineService } from '../../src/services/worker/TimelineService.js';
import { ObservationVectorIndex } from '../../src/services/vector/ObservationVectorIndex.js';
import { reciprocalRankFusion } from '../../src/services/vector/rrf.js';
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

describe('reciprocalRankFusion', () => {
  it('sums 1 / (60 + rank) across lists and keeps first-seen order on ties', () => {
    const fused = reciprocalRankFusion([
      [{ id: 1 }, { id: 2 }, { id: 3 }],
      [{ id: 3 }, { id: 4 }],
    ]);
    expect(fused.map(row => row.id)).toEqual([3, 1, 2, 4]);
  });
});

describe('SearchManager with the vector index', () => {
  function manager(vectors: ObservationVectorIndex | null): SearchManager {
    return new SearchManager(new SessionSearch(store.db), store, null,
      new FormattingService(), new TimelineService(), undefined, null, vectors);
  }

  it('fuses keyword and vector observation results', async () => {
    const keywordOnly = addObservation('p', 'Deadline fallback', 'router deadline handling');
    const meaningOnly = addObservation('p', 'Quota exhausted', 'the gateway ran out of quota');
    addObservation('p', 'Widget crash');
    await index.index(index.pendingRows(10));

    const plain = await manager(null).search({ query: 'deadline', type: 'observations', format: 'json', limit: 5 });
    expect(plain.observations.map((o: { id: number }) => o.id)).toEqual([keywordOnly]);

    // "deadline quota" has keyword hits for neither word together, but both
    // topics are near in meaning; fusion returns both.
    const fused = await manager(index).search({ query: 'deadline quota', type: 'observations', format: 'json', limit: 2 });
    expect(fused.observations.map((o: { id: number }) => o.id).sort()).toEqual([keywordOnly, meaningOnly].sort());
  });

  it('answers from keyword search alone when search by meaning fails', async () => {
    const id = addObservation('p', 'Deadline fallback');
    await index.index(index.pendingRows(10));
    embedder.fail = () => true;
    const result = await manager(index).search({ query: 'deadline', type: 'observations', format: 'json', limit: 5 });
    expect(result.observations.map((o: { id: number }) => o.id)).toEqual([id]);
  });

  it('leaves search unchanged without the index', async () => {
    addObservation('p', 'Deadline fallback');
    const result = await manager(null).search({ query: 'deadline', type: 'observations', format: 'json', limit: 5 });
    expect(result.observations).toHaveLength(1);
    expect(embedder.calls).toHaveLength(0);
  });
});

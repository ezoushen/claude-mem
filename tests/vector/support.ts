import { SessionStore } from '../../src/services/sqlite/SessionStore.js';
import type { Embedder } from '../../src/services/vector/types.js';

// One axis per topic word, so "nearest" is decided by shared topics. A text
// with none of them points along the last axis. Deterministic and offline.
export const TOPICS = ['gateway', 'quota', 'widget', 'memory', 'search', 'deadline', 'router'];

export class TopicEmbedder implements Embedder {
  readonly dims = TOPICS.length + 1;
  calls: Array<{ texts: string[]; role?: string }> = [];
  fail: ((text: string) => boolean) | null = null;
  constructor(readonly modelId = 'fake/topics') {}
  async embed(texts: string[], role?: 'query' | 'passage'): Promise<Float32Array[]> {
    this.calls.push({ texts, role });
    return texts.map(text => {
      if (this.fail?.(text)) throw new Error('cannot embed');
      const v = new Float32Array(this.dims);
      const lower = text.toLowerCase();
      TOPICS.forEach((topic, i) => { if (lower.includes(topic)) v[i] = 1; });
      if (v.every(x => x === 0)) v[this.dims - 1] = 1;
      const norm = Math.hypot(...v);
      return v.map(x => x / norm);
    });
  }
}

export interface VectorFixture {
  store: SessionStore;
  addObservation(project: string, title: string, narrative?: string | null, subtitle?: string | null): number;
}

/** An in-memory store and a helper that stores one observation in its own session. */
export function vectorFixture(): VectorFixture {
  const store = new SessionStore(':memory:');
  let sessionCounter = 0;
  return {
    store,
    addObservation(project, title, narrative = null, subtitle = null) {
      const content = `content-${++sessionCounter}`;
      const memory = `memory-${sessionCounter}`;
      const sessionId = store.createSDKSession(content, project, 'prompt');
      store.updateMemorySessionId(sessionId, memory);
      return store.storeObservation(memory, project, {
        type: 'discovery', title, subtitle, facts: [], narrative, concepts: [], files_read: [], files_modified: [],
      }, 1).id;
    },
  };
}

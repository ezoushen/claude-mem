// A stand-in for the embedding process (src/services/vector/embed-child.ts) with a tiny deterministic model:
// one axis per topic word, the same rule as support.ts's TopicEmbedder, without loading the database code.
// A query comes back at twice the length of a passage, so the test can tell which role was asked for.
// The words "crash", "hang", "fail", "partial" and "noise" make it exit, never answer, throw, die while writing
// a reply, and print stray lines
// (one of them JSON that looks like an error reply for the first request).
import { REPLY_PREFIX, runEmbeddingProcess } from '../../../src/services/vector/embed-protocol.js';
import type { Embedder } from '../../../src/services/vector/types.js';

const TOPICS = ['gateway', 'quota', 'widget', 'memory', 'search', 'deadline', 'router'];
let loaded = false;
const embedder: Embedder = {
  modelId: 'fake/topics',
  dims: TOPICS.length + 1,
  isReady: () => loaded,
  async embed(texts, role) {
    for (const text of texts) {
      if (text === 'crash') process.exit(3);
      if (text === 'hang') await new Promise(() => {});
      if (text === 'partial') {
        process.stdout.write(`${REPLY_PREFIX}{"id":0,"vectors":["AAAA`); // no end of line: killed mid-write
        process.exit(1);
      }
      if (text === 'fail') throw new Error('cannot embed');
      if (text === 'noise') {
        console.log('a line that is not a reply');
        console.log('{"id":0,"error":"stray line"}');
      }
    }
    loaded = true;
    return texts.map(text => {
      const v = new Float32Array(TOPICS.length + 1);
      TOPICS.forEach((topic, i) => { if (text.toLowerCase().includes(topic)) v[i] = 1; });
      if (v.every(x => x === 0)) v[TOPICS.length] = 1;
      const norm = Math.hypot(...v) / (role === 'query' ? 2 : 1);
      return v.map(x => x / norm);
    });
  },
};
await runEmbeddingProcess(embedder, 300);

import { afterAll, describe, expect, it, mock } from 'bun:test';

// The real module, restored after this file: bun shares modules across test
// files, so a mock left in place would fail every later real model load.
const real = { ...(await import('@huggingface/transformers')) };

// Each load records what it was asked for; the model load then stops the embed.
const loads: { kind: string; repo: string; revision: unknown }[] = [];
const recordingLoad = (kind: string) => async (repo: string, options: { revision?: string } = {}) => {
  loads.push({ kind, repo, revision: options.revision });
  if (kind === 'model') throw new Error('stop after recording the load');
  return {};
};
mock.module('@huggingface/transformers', () => ({
  ...real,
  AutoTokenizer: { from_pretrained: recordingLoad('tokenizer') },
  AutoModel: { from_pretrained: recordingLoad('model') },
}));

afterAll(() => {
  mock.module('@huggingface/transformers', () => real);
});

const { LocalEmbedder } = await import('../../../src/services/vector/LocalEmbedder.js');
const { EMBEDDING_MODELS } = await import('../../../src/services/vector/models.js');

describe('LocalEmbedder model revision', () => {
  // The Xenova/bge-m3 commit whose files were measured (2026-10-29): its
  // onnx/model_quantized.onnx has sha256 0826f8c1…0b77d, the file every q8
  // result was taken with. A later upload must not change the model under
  // stored vectors.
  it('downloads bge-m3 from the commit its results were measured on', async () => {
    loads.length = 0;
    await expect(new LocalEmbedder(EMBEDDING_MODELS['bge-m3']).embed(['a'])).rejects.toThrow('stop after recording');
    const revision = '4de13258303883538bd53b696b452bf8099f0858';
    expect(loads).toEqual([
      { kind: 'tokenizer', repo: 'Xenova/bge-m3', revision },
      { kind: 'model', repo: 'Xenova/bge-m3', revision },
    ]);
  });

  // An unpinned model keeps transformers.js's default, so its existing cache is reused.
  it('leaves an unpinned model on the default revision', async () => {
    loads.length = 0;
    await expect(new LocalEmbedder(EMBEDDING_MODELS['all-MiniLM-L6-v2']).embed(['a'])).rejects.toThrow('stop after recording');
    expect(loads).toEqual([
      { kind: 'tokenizer', repo: 'Xenova/all-MiniLM-L6-v2', revision: undefined },
      { kind: 'model', repo: 'Xenova/all-MiniLM-L6-v2', revision: undefined },
    ]);
  });
});

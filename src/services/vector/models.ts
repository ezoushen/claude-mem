/**
 * Embedding models, and what each one fixes about the stores that hold its
 * vectors.
 *
 * all-MiniLM-L6-v2 is the default because it is what Chroma used, so moving
 * off Chroma changes no ranking. It is English-only: its 30k-entry vocabulary
 * reduces Chinese, Japanese and Korean text to single characters or [UNK],
 * which is where the CJK recall gap measured in #3422 comes from (plan-25,
 * #3982). This table exists so other models, multilingual ones especially,
 * can be evaluated against it by adding an entry rather than editing the
 * embedder and the schema separately. A model's vector length is not a free
 * choice: the server's Postgres column is declared vector(dims), so the
 * schema reads it from here (see applyVectorSchema).
 *
 * CLAUDE_MEM_EMBEDDING_MODEL names the entry in use; the entries after the
 * default are the multilingual candidates measured in rich-murphey/claude-mem-search-eval.
 */
export interface EmbeddingModel {
  /** Hugging Face repository transformers.js loads. */
  repo: string;
  /** Part of the model id, so changing it re-indexes. */
  dtype: EmbeddingDtype;
  /**
   * The Hugging Face commit to download from, so a later upload to the
   * repository cannot change the model under stored vectors. Not part of the
   * model id: pin only a commit whose files match what the id was built with.
   * Unset follows the repository's main branch. A pinned commit is cached
   * apart from main, so pinning an installed model downloads it once more.
   */
  revision?: string;
  /** mean: the average over real tokens; cls: the first token's vector. */
  pooling: 'mean' | 'cls';
  /** Vector length; sizes the server's vector(n) column. */
  dims: number;
  /**
   * Tokens per input, [CLS] and [SEP] included: the length the model was
   * trained at (sentence-transformers' max_seq_length), not the tokenizer's
   * model_max_length, which is 512 for all of these.
   */
  maxTokens: number;
  /** Prepended to search queries and to stored text; e5 was trained with both. */
  queryPrefix: string;
  passagePrefix: string;
}

/** The weight variants transformers.js selects between (the model repository's onnx/ files). */
export type EmbeddingDtype = 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'q4f16' | 'bnb4';

export const EMBEDDING_MODELS = {
  'all-MiniLM-L6-v2': {
    repo: 'Xenova/all-MiniLM-L6-v2',
    dtype: 'fp32',
    pooling: 'mean',
    dims: 384,
    maxTokens: 256,
    queryPrefix: '',
    passagePrefix: '',
  },
  // The multilingual model measured in #3422.
  'paraphrase-multilingual-MiniLM-L12-v2': {
    repo: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
    dtype: 'fp32',
    pooling: 'mean',
    dims: 384,
    maxTokens: 128,
    queryPrefix: '',
    passagePrefix: '',
  },
  'multilingual-e5-small': {
    repo: 'Xenova/multilingual-e5-small',
    dtype: 'fp32',
    pooling: 'mean',
    dims: 384,
    maxTokens: 512,
    queryPrefix: 'query: ',
    passagePrefix: 'passage: ',
  },
  // IBM's granite-embedding-97m-multilingual-r2 (ModernBERT, Apache-2.0). Its
  // sentence-transformers config pools the [CLS] token and sets no prompts.
  // It was trained to 8192 tokens; cut at 512 here, because at 8192 one text
  // takes about 10 s and 8 GB on a CPU (attention grows with length squared),
  // and 512 keeps a forward pass the size of e5-small's.
  'granite-embedding-97m-multilingual-r2': {
    repo: 'onnx-community/granite-embedding-97m-multilingual-r2-ONNX',
    dtype: 'fp32',
    pooling: 'cls',
    dims: 384,
    maxTokens: 512,
    queryPrefix: '',
    passagePrefix: '',
  },
  // BAAI/bge-m3 (XLM-RoBERTa large, 568M parameters, MIT). Its dense vector is
  // the [CLS] token's, as its sentence-transformers pooling config sets, with
  // no prompts. Trained to 8192 tokens; cut at 512 for the same reason as
  // granite, and because 512 is what was measured: on a 205k-observation
  // mixed English/Chinese store it ranked best of these models, and was the
  // only one to find English notes from Chinese questions (#3610). It is
  // about twenty times e5-small's compute. Its fp32 weights are 2.3 GB; the
  // 8-bit file (q8) is 0.57 GB, takes 1.8 GB of memory once loaded (2.2 GB at
  // its peak) and is what this entry loads. Measured on the 205k-observation
  // store with note vectors made at full precision and the 44 queries embedded
  // by q8: fused nDCG@10 0.705 against 0.707, the worst query group losing
  // 0.009 (the bar was 0.02 overall, 0.05 per group), a paired per-query loss
  // of 0.002 +- 0.010. Notes embedded by q8 themselves were measured only on
  // a 11k-note subset, where they passed provisionally (overall 0.698 against
  // 0.706, natural-language queries 0.05 lower). The uint8 and int8 files are
  // as small but lost 0.016 and 0.019 overall; fp16 is exact but half the
  // speed at twice the size, and the 4-bit files are less exact (cosine 0.96
  // to fp32 against 0.98) and slower. A CPU at half its threads embedded about
  // 3 notes/s with q8 (measured on a busy machine), so a first index of a
  // large store takes many hours, and this suits a machine that can afford
  // the memory and the wait, not a default.
  'bge-m3': {
    repo: 'Xenova/bge-m3',
    dtype: 'q8',
    // Main as of 2026-02-10; its onnx/model_quantized.onnx (sha256 0826f8c1…0b77d)
    // is the file every q8 result above was measured with.
    revision: '4de13258303883538bd53b696b452bf8099f0858',
    pooling: 'cls',
    dims: 1024,
    maxTokens: 512,
    queryPrefix: '',
    passagePrefix: '',
  },
} satisfies Record<string, EmbeddingModel>;

export const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EMBEDDING_MODELS['all-MiniLM-L6-v2'];

/** The identity stored with every vector: the model's weights, pooling and token limit. */
export function embeddingModelId(model: EmbeddingModel): string {
  // The token limit is part of the identity: the same model truncated
  // elsewhere produces different vectors for long texts.
  return `${model.repo}/${model.dtype}/${model.pooling}/l2/${model.maxTokens}`;
}

/** The model a CLAUDE_MEM_EMBEDDING_MODEL value names; an unknown name is an error. */
export function resolveEmbeddingModel(name: string): EmbeddingModel {
  const model = (EMBEDDING_MODELS as Record<string, EmbeddingModel>)[name.trim()];
  if (!model) {
    throw new Error(
      `CLAUDE_MEM_EMBEDDING_MODEL="${name}" is not a known embedding model; ` +
      `use one of ${Object.keys(EMBEDDING_MODELS).join(', ')}`,
    );
  }
  return model;
}

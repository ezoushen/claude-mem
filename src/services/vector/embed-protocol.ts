import { createInterface } from 'node:readline';
import type { Embedder } from './types.js';

/**
 * The lines a worker and its embedding process exchange over stdin and stdout.
 * A request is one JSON line; a reply is one line that starts with REPLY_PREFIX
 * and then holds JSON, so anything else the process prints (a library's
 * progress output, say) is not mistaken for one.
 */
export const REPLY_PREFIX = '@embed ';

export interface EmbedRequest {
  id: number;
  texts: string[];
  role: 'query' | 'passage';
}

export type EmbedReply =
  | { id: number; vectors: string[]; loaded: boolean }
  | { id: number; error: string; loaded: boolean };

export function encodeVectors(vectors: Float32Array[]): string[] {
  return vectors.map(v => Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString('base64'));
}

export function decodeVectors(encoded: string[]): Float32Array[] {
  return encoded.map(text => {
    const bytes = Buffer.from(text, 'base64');
    // A copy: a Buffer's offset into its pool is not a multiple of four.
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  });
}

/**
 * Answer every request as it comes, without waiting for earlier ones: a search
 * must not queue behind a batch of notes being embedded (onnxruntime runs
 * concurrent calls on one session). Returns once the input has ended and every
 * request read has been answered. `loaded` on every reply says whether the
 * model had loaded by then, so a failure to load can be told from a failure to
 * embed one text. (A process that exits right after this returns can lose
 * replies still in the stdout pipe; the worker never closes the input to get
 * answers, it kills the process when it is done with it.)
 */
export async function serveEmbedder(
  embedder: Embedder,
  lines: AsyncIterable<string>,
  write: (line: string) => void,
): Promise<void> {
  const inFlight = new Set<Promise<void>>();
  for await (const line of lines) {
    if (!line.trim()) continue;
    let request: EmbedRequest;
    try {
      request = JSON.parse(line) as EmbedRequest;
    } catch {
      continue;
    }
    const answered = answer(embedder, request).then(reply => write(`${REPLY_PREFIX}${JSON.stringify(reply)}\n`));
    inFlight.add(answered);
    void answered.finally(() => inFlight.delete(answered));
  }
  await Promise.all(inFlight);
}

async function answer(embedder: Embedder, request: EmbedRequest): Promise<EmbedReply> {
  try {
    const vectors = await embedder.embed(request.texts, request.role);
    return { id: request.id, vectors: encodeVectors(vectors), loaded: embedder.isReady?.() ?? true };
  } catch (error) {
    return { id: request.id, error: error instanceof Error ? error.message : String(error), loaded: embedder.isReady?.() ?? true };
  }
}

/**
 * The body of an embedding process: serve stdin until it closes, then exit.
 * Stdin closes when the worker kills this process or dies; the process then
 * exits once the requests in progress finish, or `graceMs` later if one is
 * stuck, so it never outlives its worker for long. Replies written just before
 * the exit may not reach the pipe, which only matters to a caller that closes
 * the input and still waits for answers; the worker does not.
 */
export async function runEmbeddingProcess(embedder: Embedder, graceMs = 30_000): Promise<void> {
  process.stdin.on('end', () => {
    setTimeout(() => process.exit(0), graceMs);
  });
  await serveEmbedder(embedder, createInterface({ input: process.stdin }), line => process.stdout.write(line));
  // onnxruntime's threads would otherwise keep the process alive after its input ends.
  process.exit(0);
}

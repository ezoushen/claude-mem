import { spawn, type ChildProcess } from 'node:child_process';
import { createInterface } from 'node:readline';
import type { Embedder } from './types.js';
import { REPLY_PREFIX, decodeVectors, type EmbedReply, type EmbedRequest } from './embed-protocol.js';
import { sanitizeEnv } from '../../supervisor/env-sanitizer.js';

export interface ChildProcessEmbedderOptions {
  /** Identity of the model the process runs, as its own embedder reports it. */
  modelId: string;
  dims: number;
  /** The program and its arguments; the program serves requests with serveEmbedder. */
  command: string[];
  idleMs: number;
  requestTimeoutMs?: number;
}

interface Pending {
  resolve(vectors: Float32Array[]): void;
  reject(error: Error): void;
  timeout: ReturnType<typeof setTimeout>;
}

/** Long enough for the first request to download and load the model. */
const DEFAULT_REQUEST_TIMEOUT_MS = 10 * 60_000;

/**
 * An embedder that runs its model in a child process, started on the first
 * request and stopped after `idleMs` without one. A model held in the worker
 * keeps over a gigabyte for as long as the worker lives and cannot be given
 * back (onnxruntime keeps what it allocated after a session is released, and
 * its environment is process-wide, so a worker thread cannot own it either);
 * a process that exits returns all of it. A request after a stop starts the
 * process again, which costs the model's load time (about a second).
 */
export class ChildProcessEmbedder implements Embedder {
  readonly modelId: string;
  readonly dims: number;
  private child: ChildProcess | null = null;
  private loaded = false;
  private closed = false;
  private nextId = 0;
  private readonly pending = new Map<number, Pending>();
  private idleTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly options: ChildProcessEmbedderOptions) {
    this.modelId = options.modelId;
    this.dims = options.dims;
  }

  pid(): number | null {
    return this.child?.pid ?? null;
  }

  /** Whether the process is running with its model loaded. */
  isReady(): boolean {
    return this.child !== null && this.loaded;
  }

  async embed(texts: string[], role: 'query' | 'passage' = 'passage'): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    if (this.closed) throw new Error('Embedder closed');
    const child = this.child ?? this.start();
    this.clearIdleTimer();
    const id = this.nextId++;
    const done = new Promise<Float32Array[]>((resolve, reject) => {
      const timeout = setTimeout(() => {
        // An answer that never comes means a stuck process: end it, which fails every waiting request.
        this.fail(child, new Error(`Embedding timed out after ${this.options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS} ms`));
      }, this.options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timeout });
    });
    const request: EmbedRequest = { id, texts, role };
    child.stdin!.write(`${JSON.stringify(request)}\n`);
    return done;
  }

  private start(): ChildProcess {
    const [program, ...args] = this.options.command;
    const child = spawn(program, args, { stdio: ['pipe', 'pipe', 'inherit'], env: sanitizeEnv(process.env), windowsHide: true });
    this.child = child;
    child.stdin!.on('error', () => {}); // a process that is gone: its exit rejects the request
    child.on('error', error => this.fail(child, error));
    child.on('exit', (code, signal) => this.fail(child, new Error(`Embedding process exited (${signal ?? `code ${code}`})`)));
    createInterface({ input: child.stdout! }).on('line', line => {
      if (!line.startsWith(REPLY_PREFIX)) return;
      let reply: EmbedReply;
      try {
        reply = JSON.parse(line.slice(REPLY_PREFIX.length)) as EmbedReply;
      } catch {
        return; // the process died in the middle of a reply; its exit fails the request
      }
      const request = this.pending.get(reply.id);
      if (!request) return;
      this.pending.delete(reply.id);
      clearTimeout(request.timeout);
      this.loaded = reply.loaded;
      if ('error' in reply) request.reject(new Error(reply.error));
      else request.resolve(decodeVectors(reply.vectors));
      if (this.pending.size === 0) this.scheduleIdleStop(child);
    });
    return child;
  }

  private scheduleIdleStop(child: ChildProcess): void {
    this.clearIdleTimer();
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      if (this.child === child && this.pending.size === 0) this.fail(child, new Error('Embedding process stopped while idle'));
    }, this.options.idleMs);
    this.idleTimer.unref?.();
  }

  private clearIdleTimer(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  /** The process is gone or is to go: forget it, fail what waited on it, and kill it if it still runs. */
  private fail(child: ChildProcess, error: Error): void {
    if (this.child !== child) return;
    this.child = null;
    this.loaded = false;
    this.clearIdleTimer();
    for (const request of this.pending.values()) {
      clearTimeout(request.timeout);
      request.reject(error);
    }
    this.pending.clear();
    child.kill();
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.child) this.fail(this.child, new Error('Embedder closed'));
  }
}

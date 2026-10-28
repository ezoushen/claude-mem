import { observationDocText, type ObservationVectorIndex } from './ObservationVectorIndex.js';
import { logger } from '../../utils/logger.js';

/**
 * Keeps the observation vector index current by asking the database which
 * observations have no vector from the current model, newest first, and
 * embedding a few at a time. The same pass is the live path and the backfill,
 * so no write path has to call it, and a row whose write was missed (a
 * crash, an import, a restart mid-batch) is simply found again.
 *
 * onnxruntime runs a forward pass on the event loop's thread, so every batch
 * blocks the worker for its length. Batches are therefore small and spaced,
 * which makes a first pass over a large store with a large model slow: the
 * newest rows are embedded first so recent work becomes searchable soonest.
 */
export interface ObservationVectorIndexerOptions {
  /** Observations per batch. */
  batchSize?: number;
  /** Pause between batches while rows are pending. */
  busyDelayMs?: number;
  /** Pause between checks when nothing is pending. */
  idleDelayMs?: number;
  /** Pause after a failed batch. */
  errorDelayMs?: number;
}

const PROGRESS_LOG_EVERY = 1000;

export class ObservationVectorIndexer {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running: Promise<void> | null = null;
  private stopped = true;
  private sinceLog = 0;
  /** Observations that failed alone; skipped until the worker restarts. */
  private readonly failed = new Set<number>();
  private batchSize: number;
  private readonly fullBatchSize: number;
  private singleRuns = 0;
  private readonly busyDelayMs: number;
  private readonly idleDelayMs: number;
  private readonly errorDelayMs: number;

  constructor(private readonly index: ObservationVectorIndex, options: ObservationVectorIndexerOptions = {}) {
    this.batchSize = this.fullBatchSize = options.batchSize ?? 8;
    this.busyDelayMs = options.busyDelayMs ?? 200;
    this.idleDelayMs = options.idleDelayMs ?? 30_000;
    this.errorDelayMs = options.errorDelayMs ?? 60_000;
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    const pending = this.index.pendingCount();
    if (pending > 0) {
      logger.warn('VECTOR_INDEX', 'Embedding observations for search by meaning; a large backlog can take hours', {
        pending, indexed: this.index.indexedCount(), model: this.index.modelId,
      });
    }
    this.schedule(0);
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    await this.running;
  }

  /** One batch; resolves the number of observations taken off the queue. */
  async runOnce(): Promise<number> {
    const rows = this.index.pendingRows(this.batchSize + this.failed.size)
      .filter(row => !this.failed.has(row.id))
      .slice(0, this.batchSize);
    if (rows.length === 0) return 0;
    let written: number;
    try {
      written = await this.index.index(rows);
    } catch (error) {
      // A model that never loaded fails every row; only a loaded one blames the row.
      if (rows.length === 1 && this.index.isModelReady()) {
        // One row that cannot be embedded must not stall every row behind it.
        this.failed.add(rows[0].id);
        logger.error('VECTOR_INDEX', 'Skipping an observation that failed to embed', {
          observationId: rows[0].id,
        }, error instanceof Error ? error : new Error(String(error)));
        return 1;
      }
      throw error;
    }
    // SQLite's TRIM (spaces only) can call a note pending whose text JavaScript trims to nothing;
    // index() writes no vector for it, so it would be picked up on every pass.
    for (const row of rows) if (!observationDocText(row)) this.failed.add(row.id);
    this.sinceLog += written;
    if (this.sinceLog >= PROGRESS_LOG_EVERY) {
      logger.info('VECTOR_INDEX', 'Embedding progress', {
        indexed: this.index.indexedCount(), pending: this.index.pendingCount(),
      });
      this.sinceLog = 0;
    }
    // A batch can write nothing (every row vanished); there may still be work.
    return Math.max(written, rows.length);
  }

  private schedule(delayMs: number): void {
    if (this.stopped) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.running = this.tick().finally(() => { this.running = null; });
    }, delayMs);
    this.timer.unref?.();
  }

  private async tick(): Promise<void> {
    try {
      const done = await this.runOnce();
      // Back to full batches after a full batch's worth of single rows.
      if (this.batchSize < this.fullBatchSize && ++this.singleRuns >= this.fullBatchSize) {
        this.batchSize = this.fullBatchSize;
        this.singleRuns = 0;
      }
      this.schedule(done > 0 ? this.busyDelayMs : this.idleDelayMs);
    } catch (error) {
      // Retry the batch one row at a time, so the row at fault is isolated.
      logger.error('VECTOR_INDEX', 'Embedding batch failed; retrying row by row', {
        retryInMs: this.errorDelayMs,
      }, error instanceof Error ? error : new Error(String(error)));
      this.batchSize = 1;
      this.singleRuns = 0;
      this.schedule(this.errorDelayMs);
    }
  }
}

import { afterEach, describe, expect, it } from 'bun:test';
import { ChildProcessEmbedder } from '../../src/services/vector/ChildProcessEmbedder.js';
import { ObservationVectorIndex } from '../../src/services/vector/ObservationVectorIndex.js';
import { ObservationVectorIndexer } from '../../src/services/vector/ObservationVectorIndexer.js';
import { vectorFixture } from './support.js';

const FIXTURE = `${import.meta.dir}/fixtures/embed-child.ts`;
const DIMS = 8;
const open: ChildProcessEmbedder[] = [];
const embedder = (options: { idleMs?: number; requestTimeoutMs?: number } = {}) => {
  const e = new ChildProcessEmbedder({
    modelId: 'fake/topics', dims: DIMS, command: [process.execPath, FIXTURE], idleMs: options.idleMs ?? 60_000,
    requestTimeoutMs: options.requestTimeoutMs,
  });
  open.push(e);
  return e;
};
const alive = (pid: number) => { try { process.kill(pid, 0); return true; } catch { return false; } };
const until = async (condition: () => boolean, ms = 5000) => {
  const end = Date.now() + ms;
  while (!condition() && Date.now() < end) await new Promise(resolve => setTimeout(resolve, 20));
  return condition();
};

afterEach(async () => { await Promise.all(open.splice(0).map(e => e.close())); });

describe('ChildProcessEmbedder', () => {
  it('starts its process on the first embed, and returns what that process computed for the role asked', async () => {
    const e = embedder();
    expect(e.pid()).toBeNull();
    expect(e.isReady()).toBe(false);

    const [passage] = await e.embed(['gateway quota']);
    // Two topics: two components of 1 / sqrt(2) (fixture rule), unit length for a passage.
    expect(Array.from(passage)).toEqual(Array.from(Float32Array.from([1, 1, 0, 0, 0, 0, 0, 0].map(x => x / Math.SQRT2))));
    const [query] = await e.embed(['gateway quota'], 'query');
    expect(Math.hypot(...query)).toBeCloseTo(2, 5); // the fixture doubles a query
    expect(e.pid()).not.toBeNull();
    expect(e.isReady()).toBe(true);
  });

  it('stops its process after the idle time and starts another for the next request', async () => {
    const e = embedder({ idleMs: 150 });
    await e.embed(['widget']);
    const first = e.pid()!;
    expect(alive(first)).toBe(true);
    expect(await until(() => !alive(first))).toBe(true);
    expect(e.pid()).toBeNull();
    expect(e.isReady()).toBe(false);

    const [again] = await e.embed(['widget']);
    expect(again[2]).toBeCloseTo(1, 5);
    expect(e.pid()).not.toBe(first);
  });

  it('does not stop while a request is waiting for its answer', async () => {
    const e = embedder({ idleMs: 100, requestTimeoutMs: 600 });
    const slow = e.embed(['hang']).catch((error: Error) => error.message);
    await new Promise(resolve => setTimeout(resolve, 300)); // three idle times
    const pid = e.pid();
    expect(pid).not.toBeNull();
    expect(alive(pid!)).toBe(true);
    expect(await slow).toMatch(/timed out/);
  });

  it('answers requests made at the same time, each with its own vectors', async () => {
    const e = embedder();
    const words = ['gateway', 'quota', 'widget', 'memory', 'search', 'deadline', 'router', 'other'];
    const results = await Promise.all(words.flatMap(word => [e.embed([word]), e.embed([word], 'query')]));
    words.forEach((_, i) => {
      expect(results[2 * i][0][i]).toBeCloseTo(1, 5); // the word's own axis, unit length
      expect(results[2 * i + 1][0][i]).toBeCloseTo(2, 5); // and a query is twice that
    });
  });

  it('answers a request while an earlier one is still being computed', async () => {
    // A search must not wait behind a batch of notes the indexer sent first.
    const e = embedder({ requestTimeoutMs: 3000 });
    const slow = e.embed(['hang']).catch(() => 'timed out');
    const started = Date.now();
    const [quick] = await e.embed(['widget'], 'query');
    expect(quick[2]).toBeCloseTo(2, 5);
    expect(Date.now() - started).toBeLessThan(1000);
    expect(await slow).toBe('timed out');
  });

  it('fails the requests waiting when the process dies, and recovers on the next one', async () => {
    const e = embedder();
    await e.embed(['widget']);
    const doomed = e.pid()!;
    const waiting = [e.embed(['crash']), e.embed(['quota'])].map(p => p.then(() => 'ok', (error: Error) => error.message));
    for (const message of await Promise.all(waiting)) expect(message).toMatch(/exited/);
    expect(alive(doomed)).toBe(false);
    expect(e.isReady()).toBe(false);

    const [vector] = await e.embed(['quota']);
    expect(vector[1]).toBeCloseTo(1, 5);
    expect(e.pid()).not.toBe(doomed);
  });

  it('reports a failure to embed one text without losing the process, and says whether the model had loaded', async () => {
    const e = embedder();
    // The first request fails before the model "loads": that is a load failure, not a bad text.
    await expect(e.embed(['fail'])).rejects.toThrow('cannot embed');
    expect(e.isReady()).toBe(false);
    await e.embed(['widget']);
    const pid = e.pid();
    expect(e.isReady()).toBe(true);
    // Once loaded, the same failure is the text's: the process stays and stays ready.
    await expect(e.embed(['fail'])).rejects.toThrow('cannot embed');
    expect(e.pid()).toBe(pid);
    expect(e.isReady()).toBe(true);
    expect((await e.embed(['quota']))[0][1]).toBeCloseTo(1, 5);
  });

  it('ends a process that does not answer in time, and the next request starts a new one', async () => {
    const e = embedder({ requestTimeoutMs: 200 });
    await e.embed(['widget']);
    const stuck = e.pid()!;
    await expect(e.embed(['hang'])).rejects.toThrow(/timed out/);
    expect(await until(() => !alive(stuck))).toBe(true);
    expect((await e.embed(['quota']))[0][1]).toBeCloseTo(1, 5);
    expect(e.pid()).not.toBe(stuck);
  });

  it('ignores lines on stdout that are not replies, even ones shaped like one', async () => {
    const e = embedder();
    // The first request has id 0, and the process prints an error for id 0 that is not marked as a reply.
    expect((await e.embed(['noise']))[0][DIMS - 1]).toBeCloseTo(1, 5); // no topic word: the last axis
  });

  it('stops its process on close and refuses later requests', async () => {
    const e = embedder();
    await e.embed(['widget']);
    const pid = e.pid()!;
    await e.close();
    expect(await until(() => !alive(pid))).toBe(true);
    await expect(e.embed(['widget'])).rejects.toThrow(/closed/);
  });

  it('serves until its input ends, then exits by itself, so a worker that dies takes no process with it', async () => {
    const child = Bun.spawn([process.execPath, FIXTURE], { stdin: 'pipe', stdout: 'pipe', stderr: 'ignore' });
    child.stdin.write('{"id":1,"texts":["widget"],"role":"passage"}\n');
    child.stdin.flush();
    const reader = child.stdout.getReader();
    expect(new TextDecoder().decode((await reader.read()).value)).toStartWith('@embed {"id":1,"vectors":[');
    child.stdin.end();
    expect(await Promise.race([child.exited, new Promise(resolve => setTimeout(() => resolve('still running'), 5000))])).toBe(0);
  });
});

describe('ChildProcessEmbedder against a process that fails in other ways', () => {
  it('fails the request, and nothing else, when the process dies in the middle of a reply', async () => {
    const uncaught: unknown[] = [];
    const record = (error: unknown) => { uncaught.push(error); };
    process.on('uncaughtException', record);
    try {
      const e = embedder();
      await expect(e.embed(['partial'])).rejects.toThrow(/exited/);
      await new Promise(resolve => setTimeout(resolve, 100));
    } finally {
      process.off('uncaughtException', record);
    }
    expect(uncaught).toEqual([]);
  });

  it('exits when its input closes even while it is busy, so a killed worker leaves no stuck process', async () => {
    const child = Bun.spawn([process.execPath, FIXTURE], { stdin: 'pipe', stdout: 'ignore', stderr: 'ignore' });
    child.stdin.write('{"id":1,"texts":["hang"],"role":"passage"}\n');
    child.stdin.flush();
    await new Promise(resolve => setTimeout(resolve, 500)); // it is now inside the request
    child.stdin.end();
    expect(await Promise.race([child.exited, new Promise(resolve => setTimeout(() => resolve('still running'), 5000))])).toBe(0);
    child.kill();
  });
});

describe('ObservationVectorIndexer with a ChildProcessEmbedder', () => {
  it('skips a note the loaded model cannot embed, and blames no note while the model has not loaded', async () => {
    const fixture = vectorFixture();
    try {
      const e = embedder();
      const indexer = new ObservationVectorIndexer(new ObservationVectorIndex(fixture.store.db, e), { batchSize: 1 });
      const index = new ObservationVectorIndex(fixture.store.db, e);
      fixture.addObservation('p', 'widget');
      fixture.addObservation('p', 'quota');
      fixture.addObservation('p', 'fail'); // the newest, so the first one tried

      // The process has not loaded its model yet: the failure says nothing about the note.
      await expect(indexer.runOnce()).rejects.toThrow('cannot embed');
      expect(index.indexedCount()).toBe(0);
      await e.embed(['widget']); // now it has
      expect(await indexer.runOnce()).toBe(1); // the note is skipped...
      expect(await indexer.runOnce()).toBe(1); // ...and the others are embedded
      expect(await indexer.runOnce()).toBe(1);
      expect(index.indexedCount()).toBe(2);
    } finally {
      fixture.store.close();
    }
  });
});

describe('settings', () => {
  it('keeps the model in the worker unless unloading is asked for', async () => {
    const { SettingsDefaultsManager } = await import('../../src/shared/SettingsDefaultsManager.js');
    expect(SettingsDefaultsManager.getAllDefaults().CLAUDE_MEM_EMBEDDING_UNLOAD_AFTER_SEC).toBe('0');
  });
});

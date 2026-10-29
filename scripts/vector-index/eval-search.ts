#!/usr/bin/env bun
/**
 * Score the running worker's search on your own queries and relevance grades.
 *
 *   bun scripts/vector-index/eval-search.ts --queries queries.json --qrels qrels.json
 *       [--port N] [--max-id N] [--out run.json] [--baseline other-run.json]
 *
 * queries.json  [{ "id": "n01", "kind": "natural", "q": "why did the deploy fail last week" }, ...]
 * qrels.json    { "n01": { "18234": 2, "18240": 1 }, ... }   grade per observation id: 2 answers the
 *               query, 1 is related, anything not listed counts as 0.
 *
 * Each query goes through GET /api/search (observations only, 30 results), keeps ids up to --max-id
 * (the snapshot the grades were made on, so notes written since cannot crowd it), and scores the top
 * 10: nDCG@10, S@5 and MRR overall and per `kind`, plus latency. --baseline applies the quality bar
 * used for every variant of the local index (overall drop at most 0.02, any group at most 0.05).
 * Run it once with the index off and once with it on to compare the two on your store.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { qualityBar, scoreRun, type Qrels, type Score } from './metrics.js';

const args = process.argv.slice(2);
const option = (name: string) => { const i = args.indexOf(name); return i < 0 ? undefined : args[i + 1]; };
const queriesPath = option('--queries'), qrelsPath = option('--qrels');
if (!queriesPath || !qrelsPath) {
  console.error('usage: bun scripts/vector-index/eval-search.ts --queries queries.json --qrels qrels.json [--port N] [--max-id N] [--out run.json] [--baseline run.json]');
  process.exit(2);
}
// The worker's default port is 37700 + (uid % 100); a port set in the environment or settings needs --port.
const port = Number(option('--port') ?? process.env.CLAUDE_MEM_WORKER_PORT ?? 37700 + ((process.getuid?.() ?? 77) % 100));
const maxId = Number(option('--max-id') ?? Infinity);
const queries: { id: string; kind: string; q: string }[] = JSON.parse(readFileSync(queriesPath, 'utf8'));
const qrels: Qrels = JSON.parse(readFileSync(qrelsPath, 'utf8'));
const kinds = Object.fromEntries(queries.map(q => [q.id, q.kind]));

const run: Record<string, number[]> = {};
const ms: number[] = [];
for (const query of queries) {
  const params = new URLSearchParams({ query: query.q, limit: '30', type: 'observations', format: 'json' });
  const started = performance.now();
  const response = await fetch(`http://127.0.0.1:${port}/api/search?${params}`, { signal: AbortSignal.timeout(60_000) });
  if (!response.ok) throw new Error(`search failed for ${query.id}: HTTP ${response.status}`);
  ms.push(performance.now() - started);
  const observations: { id: number }[] = (await response.json()).observations ?? [];
  run[query.id] = observations.map(o => o.id).filter(id => id <= maxId).slice(0, 10);
}

const score = scoreRun(run, qrels, kinds);
ms.sort((a, b) => a - b);
const fmt = (s: Score) => `${s.ndcg.toFixed(3)} / ${s.s5.toFixed(2)} / ${s.mrr.toFixed(2)} (${s.n})`;
console.log('nDCG@10 / S@5 / MRR (queries)');
for (const [group, s] of Object.entries(score)) console.log(`  ${group.padEnd(12)} ${fmt(s)}`);
console.log(`latency: median ${ms[Math.floor(ms.length / 2)].toFixed(0)} ms, max ${ms[ms.length - 1].toFixed(0)} ms`);
const baselinePath = option('--baseline');
if (baselinePath) {
  const { pass, reasons } = qualityBar(JSON.parse(readFileSync(baselinePath, 'utf8')).score, score);
  console.log(pass ? 'quality bar against the baseline: pass' : `quality bar against the baseline: FAIL\n  ${reasons.join('\n  ')}`);
}
const out = option('--out');
if (out) writeFileSync(out, JSON.stringify({ run, score, ms }, null, 1));

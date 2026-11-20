#!/usr/bin/env bun
/**
 * Memory of the running worker (default port 37700 + uid % 100; use --port for another) and of any process it started (the embedding process when
 * CLAUDE_MEM_EMBEDDING_UNLOAD_AFTER_SEC is set).
 *
 *   bun scripts/vector-index/worker-memory.ts [--port N]
 *
 * On macOS it reports the physical footprint (what Activity Monitor charges the process) and its
 * peak, from vmmap. Resident size is not used: it keeps pages the allocator has freed but the kernel
 * has not taken back, which after loading the vectors overstated the worker by gigabytes. On Linux
 * it reports VmRSS and VmHWM (peak) from /proc.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const args = process.argv.slice(2);
const i = args.indexOf('--port');
const port = Number(i >= 0 ? args[i + 1] : process.env.CLAUDE_MEM_WORKER_PORT ?? 37700 + ((process.getuid?.() ?? 77) % 100));
const health = await (await fetch(`http://127.0.0.1:${port}/api/health`, { signal: AbortSignal.timeout(5000) })).json() as { pid: number; version: string };

function memory(pid: number): string {
  if (process.platform === 'darwin') {
    const out = execFileSync('vmmap', ['-summary', String(pid)], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
    const now = /Physical footprint:\s+(\S+)/.exec(out)?.[1], peak = /Physical footprint \(peak\):\s+(\S+)/.exec(out)?.[1];
    return `footprint ${now}, peak ${peak}`;
  }
  const status = readFileSync(`/proc/${pid}/status`, 'utf8');
  const field = (name: string) => /\d+/.exec(status.split('\n').find(l => l.startsWith(name)) ?? '')?.[0];
  return `VmRSS ${Math.round(Number(field('VmRSS')) / 1024)} MB, VmHWM ${Math.round(Number(field('VmHWM')) / 1024)} MB`;
}
/** Every process under `pid`, depth first, with its depth (chroma-mcp, for one, runs under uvx). */
function descendants(pid: number): { pid: number; depth: number; command: string }[] {
  let table: { pid: number; ppid: number; command: string }[];
  try {
    table = execFileSync('ps', ['-A', '-o', 'pid=,ppid=,command='], { encoding: 'utf8' }).split('\n')
      .map(l => /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(l)).filter((m): m is RegExpExecArray => !!m)
      .map(m => ({ pid: Number(m[1]), ppid: Number(m[2]), command: m[3] }));
  } catch { return []; }
  const walk = (parent: number, depth: number): { pid: number; depth: number; command: string }[] =>
    table.filter(p => p.ppid === parent).flatMap(p => [{ pid: p.pid, depth, command: p.command }, ...walk(p.pid, depth + 1)]);
  return walk(pid, 1);
}

console.log(`worker ${health.version} pid ${health.pid}: ${memory(health.pid)}`);
for (const child of descendants(health.pid)) {
  const name = child.command.includes('embed-child') ? 'embedding process'
    : child.command.includes('chroma-mcp') ? (child.command.includes('uvx') ? 'uvx for chroma-mcp' : 'chroma-mcp')
      : child.command.split(' ').slice(0, 2).join(' ');
  let usage: string;
  try { usage = memory(child.pid); } catch { usage = 'gone'; }
  console.log(`${'  '.repeat(child.depth)}child ${child.pid} (${name}): ${usage}`);
}

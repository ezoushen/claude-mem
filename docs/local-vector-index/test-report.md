# Test report: the local vector index against Chroma

Measured 2026-10-29 on one store. Three setups, each run as its own worker on its own copy of the same
notes and scored on the same 42 graded queries:

- **Chroma:** stock claude-mem 15.5.0 with Chroma on, at its defaults: local mode and the default embedding
  function, so `all-MiniLM-L6-v2`.
- **Branch, default model:** this branch with the index on and `CLAUDE_MEM_EMBEDDING_MODEL` left at its
  default, `all-MiniLM-L6-v2` at fp32, the model Chroma uses. The two download different ONNX exports of it
  (same size, 90,387,606 bytes, different contents).
- **Branch, default model, as run:** with `CLAUDE_MEM_EMBEDDING_UNLOAD_AFTER_SEC=300`, so the model ran in a
  separate process that exits after 5 idle minutes. The shipped default, 0, keeps the model in the worker;
  that setting was measured too (see [Memory](#memory)) and ranks identically.
- **Branch, bge-m3:** this branch with `CLAUDE_MEM_EMBEDDING_MODEL=bge-m3`, the setup the
  [guide](README.md) recommends.

Keyword-only search (stock 15.5.0 with Chroma off) is the baseline.

## Summary

| | Keyword only | Chroma | Branch, default model (process exits when idle) | Branch, bge-m3 (the same) |
| --- | --- | --- | --- | --- |
| nDCG@10 (42 queries) | 0.396 | 0.394 | **0.531** | **0.734** |
| Chinese queries naming terms in the notes | 0.441 | 0.041 | 0.546 | 0.959 |
| Chinese questions about English notes | 0.000 | 0.000 | 0.000 | 0.660 |
| Search, median | 16 to 29 ms | 110 ms | 19 to 32 ms | 105 ms |
| Memory while searching | worker only | worker 75 to 98 MB, plus `chroma-mcp` 0.39 to 0.65 GB | worker 130 MB, plus embedding process 233 MB | worker 155 to 175 MB, plus embedding process 1.6 GB |
| Memory while idle | worker only | the same: `chroma-mcp` stays up | worker 114 MB (model kept in the worker: 276 to 292 MB) | worker 155 MB |
| Index on disk, 11k notes | none | 309 MB | 24 MB | about 52 MB |
| Index on disk, 216k notes | none | 6.9 GB (production folder, when deleted) | 0.46 GB | 1.01 GB |
| Indexing 11k notes on the CPU | none | about 30 min | 8 min 20 s | about 1 h (not timed here) |
| Model download | none | 83 MB compressed (91 MB unpacked), plus Python and `uv` | 91 MB | 587 MB |
| Extra processes | none | `uvx`, `chroma-mcp` (Python) | one embedding process, only while in use | the same |

- **Same model, different design.** With the same model as Chroma, the branch scores 0.531 against
  Chroma's 0.394. The branch is better on 24 queries and worse on 10 (mean +0.137 ± 0.048).
- **The default setup is the cheapest way to search by meaning.** It indexes 3.6 times faster than Chroma, uses less memory in total
  while searching (about 0.36 GB against 0.5 to 0.9 GB, or 0.29 GB with the model kept in the worker) and a
  thirteenth of the disk. The default model alone is reason enough to replace Chroma.
- **bge-m3 adds another +0.203 ± 0.050.** It is the only setup that answers Chinese questions about English
  notes. It costs about twice the disk of the default model, a 6.5 times larger download and 1.6 GB of
  memory while its process runs.
- **Chroma gains nothing overall over keyword search** (−0.002 ± 0.055). It helps English queries (agent
  +0.11, natural +0.12) and loses Chinese term searches (0.441 to 0.041), which fails the quality bar.
- **A correction to earlier figures.** An earlier version of these results gave Chroma 0.376 against
  keyword's 0.401. About a third of Chroma's top results had never been graded and counted as irrelevant (see
  [Grading](#grading)). With them graded, Chroma ties with keyword search rather than falling below it.

## The store and the queries

- **The store.** 11,056 observations from one owner's store: every graded note still in the store (29
  graded notes had been deleted, all graded 0), plus 10,000 chosen at random (seed 20261029) from the
  215k-note store (claude-mem 15.4.0 to 15.5.0, mostly English with about 700 Chinese notes). Summaries, prompts and queues were emptied, so search covers observations
  only. On this subset keyword search scores the same as on the full store (0.40). The copy was rebuilt
  from the live database for the default-model run. Keyword search on it gave exactly the same results
  as on the copy Chroma and bge-m3 ran on (0.401 / 0.64 / 0.45 under the original grades, every group
  equal), so it holds the same notes.
- **The queries.** 44 queries, 42 of which have a note graded 2 and are scored. The 42 come in four
  groups:
  - `agent` (15): searches agents actually ran.
  - `natural` (12): plain-English questions.
  - `zh-term` (8): Chinese queries naming terms in the notes.
  - `zh-cross` (7): Chinese queries about English notes.

  The queries describe private work and are not published.
- **Scoring.** `scripts/vector-index/eval-search.ts` sends each query to the worker's `/api/search`
  (observations, 30 results) and scores the top 10:
  - nDCG@10, with gain 2^grade − 1.
  - S@5: the share of queries with a grade-2 note in the top 5.
  - MRR.

  The quality bar used throughout this work: overall nDCG@10 may drop at most 0.02, and no group may drop
  more than 0.05.
- **The machine.** Apple M5 Max, 18 cores, 128 GB, macOS. It was busy with other work during every run: load
  average 5 to 20 while the default model indexed. Speeds are lower bounds.

### Grading

Notes are graded 2 (answers the query), 1 (related) or 0 (unrelated) by Claude Sonnet judges. Each judge
sees the query and the note's project, title, subtitle and first 400 characters, shuffled, with the method
hidden.

- **The first pass (2026-10-28)** pooled the top 10s of the offline methods. Chroma's and the live
  branch's results were not in that pool: over the 42 scored queries, 31 % of Chroma's top 10s and 3 % of
  the branch's had no grade and counted as 0. Scored that way, Chroma looked worse than it is.
- **This report** graded the 179 missing pairs in a second pass with the same instructions. Every top 10
  scored here now has a grade.
- **Checking the second pass.** Each of the 36 queries in the second pass also got three notes the first
  pass had graded, unmarked, as anchors. On those 108 the two passes agree exactly on 71 % and within one
  grade on 99 % (linear-weighted kappa 0.62). The second pass is the more lenient: it raised 29 grades (20
  from 0 to 1, 8 from 1 to 2, 1 from 0 to 2) and lowered 2. Most newly graded notes are Chroma's (139 of
  163 on scored queries), so if anything Chroma's score is now overstated.

Under the first-pass grades alone, the scores were: keyword 0.401, Chroma 0.376 (window off 0.373), branch
with the default model 0.533 and branch with bge-m3 0.739.

## Quality

nDCG@10 / S@5 / MRR, by query group, on the completed grades.

| Setup | All (42) | agent (15) | natural (12) | zh-term (8) | zh-cross (7) |
| --- | --- | --- | --- | --- | --- |
| Keyword only | 0.396 / 0.64 / 0.45 | 0.553 / 0.93 / 0.71 | 0.401 / 0.75 / 0.44 | 0.441 / 0.50 / 0.37 | 0.000 / 0.00 / 0.00 |
| Chroma, default 90-day window | 0.394 / 0.55 / 0.37 | 0.666 / 0.93 / 0.61 | 0.519 / 0.75 / 0.54 | 0.041 / 0.00 / 0.02 | 0.000 / 0.00 / 0.00 |
| Chroma, window off | 0.390 / 0.55 / 0.35 | 0.660 / 0.93 / 0.59 | 0.479 / 0.75 / 0.46 | 0.088 / 0.00 / 0.04 | 0.000 / 0.00 / 0.00 |
| Branch, default model | 0.531 / 0.74 / 0.55 | 0.731 / 0.93 / 0.75 | 0.582 / 0.92 / 0.63 | 0.546 / 0.75 / 0.57 | 0.000 / 0.00 / 0.00 |
| Branch, bge-m3 | 0.734 / 0.86 / 0.79 | 0.728 / 0.87 / 0.72 | 0.636 / 0.83 / 0.81 | 0.959 / 1.00 / 1.00 | 0.660 / 0.71 / 0.68 |

Per query (nDCG@10 of A minus B, mean ± standard error over the 42 queries):

| A against B | Mean difference | A better / tie / B better | Quality bar, B as baseline |
| --- | --- | --- | --- |
| Branch, default model against keyword only | +0.135 ± 0.030 | 27 / 10 / 5 | passes |
| Branch, default model against Chroma (90 days) | +0.137 ± 0.048 | 24 / 8 / 10 | passes |
| Branch, default model against Chroma (window off) | +0.142 ± 0.043 | 25 / 8 / 9 | passes |
| Branch, bge-m3 against branch, default model | +0.203 ± 0.050 | 27 / 2 / 13 | passes |
| Chroma (90 days) against keyword only | −0.002 ± 0.055 | 17 / 10 / 15 | fails: zh-term −0.400 |

The default model was scored twice: once with its model in a separate process and once with the model
inside the worker. Both gave the same ranking.

### Where the difference comes from

Chroma and the branch's default setup run the same model, so the +0.137 between them comes from how the
results are used, not from the model:

- **Fusion with keyword search.** The branch fuses keyword and vector results by reciprocal rank. With
  Chroma on, stock search ranks by Chroma's results instead. In the offline evaluation on the full store
  (2026-10-28, below), fusion took MiniLM from 0.39 to 0.49. Fusion is the likely main cause, but that
  figure comes from another store size and setup, and this test does not isolate it.
- **Chinese term searches** show it most clearly. Keyword search finds them (0.441), Chroma's English-only
  model does not (0.041), and the branch keeps the keyword hits (0.546).
- **One document per note.** Chroma splits each note into several documents (narrative, text, each fact):
  62,158 for these 11,056 notes. The branch embeds one per note, from the title, subtitle and narrative.
  This test does not separate this effect from fusion.

From the default model to bge-m3 the gain is the model's: bge-m3 is multilingual and much larger (568M
parameters against 22M). Chinese questions about English notes go from 0.000 to 0.660.

## Speed

### Search

Time for one `/api/search` call over the 44 queries, in ms: median / 90th percentile / slowest. The
keyword run beside Chroma (stock build, earlier in the day) took 29 / 75 / 512; the one listed ran on the
branch build with the index off.

| Setup | Times | Slowest, and why |
| --- | --- | --- |
| Keyword only | 16 / 28 / 43 | |
| Chroma (90 days) | 110 / 115 / 127 | |
| Branch, default model, separate process | 19 / 29 / 92; a second pass 25 median | 145 ms for a search after the process had exited when idle |
| Branch, default model, model in the worker | 32 median, then 30 on a second pass (72 slowest) | 799 ms; the worker had just started, so one search loaded the model |
| Branch, bge-m3, separate process | 105 / 174 / 1,548 | 1,548 ms; the run began with no embedding process, so one search started it and loaded the model |

### Indexing

On the CPU, as each worker runs it (the branch uses half the cores, 9 threads):

| Setup | 11,056 notes | Rate | 216k notes |
| --- | --- | --- | --- |
| Chroma | about 30 min, 62,158 documents | about 34 documents (6 notes) a second | 12 to 15 hours (25 to 31 documents a second, production) |
| Branch, default model | 8 min 20 s | 22 notes a second, at load average 5 to 20 | about 2.7 hours (estimate from the rate) |
| Branch, bge-m3 | not timed here (imported) | 3.3 notes a second (busy machine, 2026-10-29) | most of a day on the CPU; 23 minutes with the GPU script in the guide |

## Memory

macOS physical footprint, from `scripts/vector-index/worker-memory.ts` (vmmap), with the peak in
brackets. Resident size overstates these processes and is not used.

| Setup | Indexing | Searching | Idle |
| --- | --- | --- | --- |
| Chroma | not sampled | worker 75 to 98 MB (159 MB); `chroma-mcp` 0.39 to 0.65 GB (0.82 GB); `uvx` 23 to 143 MB | the same; `chroma-mcp` stays up |
| Branch, default model, separate process | worker 66 to 77 MB; embedding process 232 MB (297 MB) | worker 130 MB (145 MB); embedding process 233 to 289 MB (301 MB) | worker 114 MB, 5 minutes after the last request |
| Branch, default model, model in the worker | | worker 292 MB (381 MB) | the same; the model stays loaded |
| Branch, bge-m3, separate process | | worker 155 to 175 MB (0.20 GB); embedding process 1.6 GB (2.1 GB while loading) | worker 155 MB |

The idle time is `CLAUDE_MEM_EMBEDDING_UNLOAD_AFTER_SEC`, set to 300 for these runs. With the default, 0,
the model stays in the worker.

## Disk

| | 11,056 notes | 216k notes |
| --- | --- | --- |
| Chroma | 309 MB (the `chroma` folder) | 6.9 GB freed when the production folder was deleted (2026-10-28); 5.1 GB on 2026-10-25; 1.7 GB at 369,641 documents (2026-09-15) |
| Branch, default model | 24 MB (`vec_observation_docs` with its indexes) | 0.46 GB |
| Branch, bge-m3 | about 52 MB | 1.01 GB (996 MB table, 16 MB indexes) |

- **Branch, default model, 216k notes:** a copy of the live table with every row's ids and hashes and a
  random 1,536-byte vector (384 float32 values) in place of the 4,096-byte bge-m3 one. Vectors have a fixed
  length, and the rows carry the MiniLM model id, so the size is exact: 444 MB of table and 17 MB of
  indexes.
- **Branch, bge-m3, 11k notes:** scaled from the full-store table (4.7 KB per row); the vectors themselves
  are 45 MB.
- **Chroma's size** depends on more than the vectors. It stores each document's text and metadata and an
  HNSW graph, about 5 KB per document, and its production folder grew by 1.8 GB in its last three days.

The model files are separate: 91 MB for MiniLM (Chroma downloads its own export as an 83 MB archive), and
587 MB for bge-m3. Chroma also needs Python and a `uv` environment for `chroma-mcp`.

## The full store

The 11k-note copy was used because indexing all 216k notes with Chroma would take 12 to 15 hours. At the
full store:

- **Quality**, from the offline evaluation of 2026-10-28: the same 42 queries over a 205k-note snapshot,
  with sentence-transformers models and first-pass grades:

  | Method | nDCG@10 |
  | --- | --- |
  | Keyword only | 0.40 |
  | MiniLM alone | 0.39 |
  | Keyword + MiniLM | 0.49 |
  | bge-m3 alone | 0.66 |
  | Keyword + bge-m3 | 0.71 |

  The installed branch with bge-m3 scores 0.69 to 0.70 on the live store. The 11k copy gives the vector
  setups higher scores than the full store does (0.531 against 0.49, 0.734 against 0.71), because there
  are likely fewer other notes to compete with (the setups and grades also differ). Compare setups
  within one store size, not across sizes.
- **Disk:** as in the table above.
- **Memory for search:** the branch keeps one int8 row per note in memory, plus a 4-byte scale and a
  4-byte id.
  - Default model: 85 MB (216,140 × 392 bytes).
  - bge-m3: 223 MB (216,140 × 1,032 bytes).
  - Chroma: 1.35 million 384-dimension vectors, 2.1 GB as float32, which its HNSW index keeps in memory
    while it runs.

  These are computed, not measured. With bge-m3 the measured worker was 0.4 GB (0.9 GB peak).

## How to repeat this

The [guide](README.md#reproduce-the-measurements) has the commands. To run a test worker beside your own
on a copy of the store:

```bash
sqlite3 ~/.claude-mem/claude-mem.db "VACUUM INTO '/path/to/copy/claude-mem.db'"
cd <a copy of the built plugin folder>
DO_NOT_TRACK=1 CLAUDE_MEM_DATA_DIR=/path/to/copy CLAUDE_MEM_WORKER_PORT=37792 bun scripts/worker-service.cjs --daemon
bun scripts/vector-index/eval-search.ts --queries queries.json --qrels qrels.json --port 37792 --out run.json
bun scripts/vector-index/worker-memory.ts --port 37792
```

The test worker reads `settings.json` from its data folder. Turn off everything in it that acts outside
the copy or runs an agent, or it will act for you:

- `CLAUDE_MEM_WORK_VIEW_AUTO_REFRESH=false`. Its scheduled refresh starts a Claude Code, Codex or Gemini run
  on your login; a test worker for this report started one before it was turned off.
- `CLAUDE_MEM_TRANSCRIPTS_ENABLED=false`.
- `CLAUDE_MEM_TELEGRAM_ENABLED=false`.
- Cloud sync: empty `CLAUDE_MEM_CLOUD_SYNC_TOKEN` and `CLAUDE_MEM_CLOUD_SYNC_HUB_URL`.
- Empty provider API keys.
- `telemetry.json` with `"enabled": false`.

Grade every note any compared setup returns in its top 10, not only the ones in your first pool.

## Limits

- **One store, one owner, 42 graded queries,** with 7 to 15 per group. Differences of a few hundredths are
  noise; the standard errors above show how much.
- **Model judges, one pass per note.** No human checked the grades. The two passes agree within one grade
  on 99 % of the anchors, but the second is more lenient. A relevant note that no setup returned is never
  graded.
- **Chroma was run at its defaults** and not tuned. Chroma's memory while indexing was not sampled.
- **Full-store figures for Chroma** come from the production install it replaced; whether that index was
  complete when it was deleted is not known.
- **Not measured:** the branch with the default model on the full store (estimated above), and Linux and
  Windows.

# Local vector index: search by meaning without Chroma

This branch (`local-vector-index` on [ezoushen/claude-mem](https://github.com/ezoushen/claude-mem)) gives
the claude-mem worker search by meaning from vectors stored in `claude-mem.db` itself. It needs no Python,
no `uv` and no `chroma-mcp` process. The model is `BAAI/bge-m3`, run in-process by transformers.js and
onnxruntime-node, which the plugin already ships. Keyword results and vector results are fused by
reciprocal rank.

It is opt-in. With `CLAUDE_MEM_VECTOR_INDEX_ENABLED` unset or `false`, the worker behaves like stock
15.5.0: no table, no model, no timers, the same search results.

This page argues that this path should replace Chroma, gives the measurements behind that, and shows how
to set it up and reproduce the measurements on your own store.

- [Why replace Chroma](#why-replace-chroma)
- [Results](#results)
- [Set up](#set-up)
- [Fill the index faster on Apple Silicon](#fill-the-index-faster-on-apple-silicon)
- [Check it works](#check-it-works)
- [Reproduce the measurements](#reproduce-the-measurements)
- [Turn it off](#turn-it-off)
- [Limits](#limits)

## Why replace Chroma

- **Quality.** Chroma's default embedding function, and PR #3694, use `all-MiniLM-L6-v2`. On a real store
  it scored no better than keyword search (nDCG@10 0.39 against 0.40) and found nothing for a question in
  one language about notes in another. bge-m3 fused with keyword search scored 0.71, and 0.62 on
  cross-language questions.
- **Moving parts.** Chroma needs Python, `uv` and a `chroma-mcp` process beside the worker, plus a second
  store to keep in step. Here the vectors are rows in `claude-mem.db`, with a foreign key to their
  observation, so a deleted observation takes its vector with it.
- **Memory.** On a 214k-observation store the worker holds about 1.1 GB with the model loaded, or about
  0.5 GB when the model runs in a separate process that exits when idle (see below).

The numbers come from one store (claude-mem 15.4.0 to 15.5.0, 205k to 214k observations, mostly English
with about 700 Chinese notes, Apple M5 Max with 128 GB). The queries and grades describe private work, so
they are not published; the method and the tools are, so you can measure your own store the same way.
Chroma itself was not measured: the comparison is with keyword search, which is what the worker does with
Chroma off, and with MiniLM, the model Chroma embeds with by default.

## Results

### Which model (2026-10-28, offline, 42 judged queries)

nDCG@10 / S@5 / MRR. `agent`: searches agents actually ran; `natural`: plain-English questions;
`zh-term`: Chinese queries naming terms in the notes; `zh-cross`: Chinese queries about English notes.

| Method | All (42) | agent (15) | natural (12) | zh-term (8) | zh-cross (7) |
| --- | --- | --- | --- | --- | --- |
| keyword (worker today) | 0.40/0.64/0.45 | 0.55/0.93/0.71 | 0.43/0.75/0.44 | 0.44/0.50/0.37 | 0.00/0.00/0.00 |
| MiniLM (Chroma default, #3694) | 0.39/0.55/0.43 | 0.72/1.00/0.81 | 0.33/0.50/0.31 | 0.22/0.25/0.27 | 0.00/0.00/0.00 |
| multilingual-e5-small | 0.52/0.71/0.59 | 0.71/0.93/0.76 | 0.50/0.75/0.60 | 0.56/0.75/0.75 | 0.09/0.14/0.05 |
| bge-m3 | 0.66/0.79/0.70 | 0.71/0.87/0.74 | 0.55/0.67/0.60 | 0.75/0.88/0.88 | 0.62/0.71/0.57 |
| **keyword + bge-m3 (this branch)** | **0.71/0.86/0.74** | 0.71/0.87/0.72 | 0.60/0.83/0.70 | 0.93/1.00/1.00 | 0.62/0.71/0.57 |

Method: the top 10 of each method were pooled per query, shuffled with the method hidden, and graded 0, 1
or 2 by four Claude Sonnet judges; nDCG uses gain 2^grade - 1; S@5 is the share of queries with a grade-2
note in the top 5; MRR is of the first grade-2 note. Queries with no grade-2 note (2 of 44) are left out.

### The installed worker (2026-10-29, the same queries through `/api/search`)

| Worker | All | agent | natural | zh-term | zh-cross | S@5 |
| --- | --- | --- | --- | --- | --- | --- |
| stock 15.5.0, keyword only | 0.403 | 0.552 | 0.428 | 0.441 | 0.000 | |
| this branch | 0.700 | 0.689 | 0.608 | 0.929 | 0.618 | 0.83 |

Median search latency through the API: 264 ms against 229 ms for keyword only.

### Which weights (the query model, 205k-note store, note vectors at full precision)

The bar every cheaper variant had to clear: fused nDCG@10 no more than 0.02 below full precision overall,
and no group more than 0.05 below.

| ONNX file | Size | Fused nDCG@10 | Loss (paired, SE) | Bar |
| --- | --- | --- | --- | --- |
| fp32 | 2.27 GB | 0.707 | | baseline |
| **q8 (used)** | 0.57 GB | 0.705 | 0.002 +- 0.010 | pass |
| uint8 | 0.57 GB | 0.692 | 0.016 +- 0.012 | pass, inside its noise |
| int8 | 0.57 GB | 0.688 | 0.019 +- 0.010 | pass, inside its noise |
| q4 | 1.25 GB | 0.680 | 0.027 +- 0.010 | fail |
| q4f16 | 0.70 GB | 0.680 | 0.027 +- 0.010 | fail |
| bnb4 | 1.23 GB | 0.662 | 0.045 +- 0.014; zh-cross -0.157 | fail |

The 4-bit files also load into more memory than q8 (2.3 to 4.1 GB against 1.85 GB resident) and run two to
three times slower on the CPU, so 8-bit is the floor.

### How the vectors are held in memory (same store and queries, fp32 queries)

| In memory | Size at 213k notes | Fused nDCG@10 | Top 10 identical to float32 |
| --- | --- | --- | --- |
| float32 | 870 MB | 0.707 | |
| float16 | 435 MB | 0.708 | 44 of 44 |
| int8, one scale per vector | 218 MB | 0.703 | 99.1 % |
| **int8, best candidates re-scored from the stored float32 (used)** | 218 MB | 0.707 | 44 of 44 |
| one bit per dimension, best 100 to 3,000 re-scored | 27 MB | 0.64 to 0.66 | fails the bar |

### Worker memory (macOS physical footprint, the live worker)

| Build | Worker | Peak |
| --- | --- | --- |
| float32 vectors, model in the worker | 1.8 GB | 4.5 GB |
| int8 vectors, model in the worker | 1.1 GB | 2.2 GB |
| int8 vectors, model in a process that exits when idle | 0.5 GB, plus 1.0 to 1.3 GB for the embedding process while it runs (2.1 GB at its peak, while the model loads) | 0.9 GB |

The resident size `ps` shows was 4.6 GB for the first build: it counts pages freed after loading that the
kernel had not taken back. Measure footprint instead (`scripts/vector-index/worker-memory.ts`).

### Indexing time for 212k notes

| Route | Speed | Time |
| --- | --- | --- |
| the worker, CPU, half the threads (q8) | about 3 notes/s | most of a day; newest notes first |
| `embed_store_mlx.py`, Apple GPU (MLX fp16) | 154 notes/s | 23 minutes |

Vectors made on the GPU match the worker's full-precision ONNX vectors (cosine 1.0000 on 2,000 notes).

## Set up

You need claude-mem 15.5.0 installed in Claude Code, and Bun.

**1. Build the branch.**

```bash
git clone -b local-vector-index https://github.com/ezoushen/claude-mem.git
cd claude-mem
bun install
npm run build
```

**2. Install it over your plugin.** Find the plugin folder, keep a copy of what you replace, and copy the
files this branch changes (the built bundles and the new embedding-process script):

```bash
# The list of changed files is taken against the v15.5.0 tag; fetch it if your clone lacks it.
git rev-parse -q --verify v15.5.0 >/dev/null || git fetch https://github.com/thedotmack/claude-mem.git tag v15.5.0
PLUGIN=$(jq -r '.plugins | to_entries[] | select(.key | test("claude-mem")) | .value[0].installPath' \
  ~/.claude/plugins/installed_plugins.json)
mkdir -p ~/claude-mem-backup
git diff --name-only v15.5.0..HEAD -- plugin | sed 's#^plugin/##' | while read -r f; do
  if [ -f "$PLUGIN/$f" ]; then mkdir -p ~/claude-mem-backup/$(dirname "$f"); cp -p "$PLUGIN/$f" ~/claude-mem-backup/"$f"; fi
  mkdir -p "$PLUGIN/$(dirname "$f")"; cp -p "plugin/$f" "$PLUGIN/$f"
done
```

Turn off automatic updates for the claude-mem marketplace while you run a local build, or an update will
replace it: set `"autoUpdate": false` for `thedotmack` in `~/.claude/plugins/known_marketplaces.json`.

**3. Settings.** In `~/.claude-mem/settings.json`:

```json
{
  "CLAUDE_MEM_CHROMA_ENABLED": "false",
  "CLAUDE_MEM_VECTOR_INDEX_ENABLED": "true",
  "CLAUDE_MEM_EMBEDDING_MODEL": "bge-m3",
  "CLAUDE_MEM_EMBEDDING_UNLOAD_AFTER_SEC": "300"
}
```

| Setting | Meaning |
| --- | --- |
| `CLAUDE_MEM_VECTOR_INDEX_ENABLED` | `true` turns the index on. Ignored, with a warning, while Chroma is on. |
| `CLAUDE_MEM_EMBEDDING_MODEL` | `bge-m3` for the results above. `all-MiniLM-L6-v2` (the default) and the other models in `src/services/vector/models.ts` also work. |
| `CLAUDE_MEM_EMBEDDING_UNLOAD_AFTER_SEC` | `0` (default) keeps the model in the worker. A number of seconds runs it in `scripts/embed-child.cjs`, which starts on demand and exits after that long without a request; the first search after that waits about a second. |
| `CLAUDE_MEM_EMBEDDING_BACKFILL_THREADS` | CPU threads for the model. Empty is half of them, `0` is all. |

**4. The model.** The first vector request downloads it from Hugging Face into the plugin's
transformers.js cache, `$PLUGIN/node_modules/@huggingface/transformers/.cache/Xenova/bge-m3/`: four files,
587 MB (`onnx/model_quantized.onnx` 570 MB, `tokenizer.json` 17 MB, `tokenizer_config.json`, `config.json`).
To place them yourself, for a machine without access to Hugging Face:

```bash
hf download Xenova/bge-m3 onnx/model_quantized.onnx tokenizer.json tokenizer_config.json config.json \
  --local-dir "$PLUGIN/node_modules/@huggingface/transformers/.cache/Xenova/bge-m3"
```

(`hf` comes with `pip install huggingface_hub`; older versions call it `huggingface-cli download`.)

**5. Restart the worker.**

```bash
cd "$PLUGIN" && bun scripts/worker-service.cjs restart
```

The worker now embeds every observation in the background, newest first, and fuses search results as
soon as any are embedded. On a large store that takes many hours on the CPU; the next section does it in
minutes on a Mac.

## Fill the index faster on Apple Silicon

`scripts/vector-index/embed_store_mlx.py` embeds every observation with bge-m3 on the GPU, reading the
database read-only and writing only vectors and ids. `scripts/vector-index/import-vectors.ts` stores them
under the model identity the worker uses, so the worker treats them as done.

```bash
python3 -m venv ~/bge-m3-venv && ~/bge-m3-venv/bin/pip install -r scripts/vector-index/requirements-mlx.txt
~/bge-m3-venv/bin/python scripts/vector-index/embed_store_mlx.py run --out ~/bge-m3-store       # safe to stop and rerun
~/bge-m3-venv/bin/python scripts/vector-index/embed_store_mlx.py assemble --out ~/bge-m3-store

sqlite3 ~/.claude-mem/claude-mem.db ".backup '$HOME/claude-mem-before-import.db'"
bun scripts/vector-index/import-vectors.ts ~/.claude-mem/claude-mem.db \
  ~/bge-m3-store/store.f32 ~/bge-m3-store/store.ids.json --dry    # what it would write
bun scripts/vector-index/import-vectors.ts ~/.claude-mem/claude-mem.db \
  ~/bge-m3-store/store.f32 ~/bge-m3-store/store.ids.json
```

The import can run while the worker runs (it writes in short transactions), and again later: vectors already
stored are left alone. Notes written after the GPU run are embedded by the worker itself. The imported
vectors are full precision and are stored under the q8 identity; that is the combination measured above
(0.705), and better than letting the CPU make q8 vectors for every note.

The first run downloads `mlx-community/bge-m3-mlx-fp16` (1.1 GB). On other hardware, any bge-m3 vectors
made the same way can be written as `store.f32` and `store.ids.json` and imported the same way: the `[CLS]`
state, 512 tokens, unit length, over the worker's document text (title, subtitle, and narrative or else the
legacy `text`, each trimmed, empty ones dropped, joined by newlines; see `observationDocText` in
`src/services/vector/ObservationVectorIndex.ts`).

## Check it works

- The worker log (`~/.claude-mem/logs/`) shows `Search by meaning from the local vector index`, then
  `Embedding progress {indexed, pending}` while it fills, and `Loaded vectors for search {vectors,
  megabytes}` at the first search.
- `bun scripts/vector-index/worker-memory.ts` prints the worker's memory, and the embedding process's when
  one runs.
- A search for something you wrote in other words, or in another language, now finds it.

## Reproduce the measurements

**Quality on your store.**

1. Write 30 to 50 queries as `queries.json`: `[{ "id": "n01", "kind": "natural", "q": "..." }]`. Mix the
   kinds you care about: searches your agents actually ran (in your session logs), plain questions, other
   languages.
2. Collect candidates: run them with the index off and on (below), and any other method you want to compare;
   take the top 10 of each per query.
3. Grade every candidate 0, 1 or 2 without seeing which method found it, into
   `qrels.json`: `{ "n01": { "18234": 2, "18240": 1 } }`. The results above used four Claude Sonnet judges on
   the note titles and subtitles.
4. Score the worker both ways, noting the highest observation id when you grade so later notes do not count:

```bash
# CLAUDE_MEM_VECTOR_INDEX_ENABLED=false, worker restarted:
bun scripts/vector-index/eval-search.ts --queries queries.json --qrels qrels.json --max-id 214000 --out keyword.json
# CLAUDE_MEM_VECTOR_INDEX_ENABLED=true, worker restarted, index filled:
bun scripts/vector-index/eval-search.ts --queries queries.json --qrels qrels.json --max-id 214000 --out vectors.json \
  --baseline keyword.json
```

`scripts/vector-index/metrics.ts` holds the scoring (nDCG@10, S@5, MRR and the quality bar), with tests in
`tests/vector/search-eval-metrics.test.ts`. Both scripts talk to the worker's default port, 37700 + (your uid
% 100); pass `--port` if you set another.

**Memory.** `bun scripts/vector-index/worker-memory.ts` after a few searches, with
`CLAUDE_MEM_EMBEDDING_UNLOAD_AFTER_SEC` at `0` and at `300`.

**Tests.** `bun test tests/vector tests/services/vector` covers the index, the fusion, the 8-bit matrix
with its re-score, the embedding process (against a real child process) and the import.

## Turn it off

Set `CLAUDE_MEM_VECTOR_INDEX_ENABLED` to `false` and restart: search is keyword-only again, exactly as
stock. The vectors stay in `claude-mem.db` (about 4 KB per observation with bge-m3) until you drop them:

```bash
sqlite3 ~/.claude-mem/claude-mem.db "DROP TABLE vec_observation_docs"
```

To go back to stock completely, copy `~/claude-mem-backup` over the plugin folder, turn automatic updates
back on and restart.

## Limits

- One store, one owner, 42 graded queries with 7 to 15 per group: differences of a few hundredths are noise.
  Grading was pooled (a relevant note no method found is never graded) and done by a model.
- Only observations are embedded; summaries and prompts stay keyword-only. One document per observation
  (title, subtitle and narrative), where Chroma and #3694 split notes into several.
- Filling the index on the CPU takes hours for a large store; the GPU script is for Apple Silicon only.
- Searches take about 90 ms longer than with float32 vectors, for the int8 matrix's conversion.
- The embedding process is tested on macOS; on Windows it is untested (it is spawned hidden, like the
  worker's other children).
- A note that makes the model crash the embedding process (rather than fail) is retried every minute and
  holds back older notes until it is fixed; nothing is lost.

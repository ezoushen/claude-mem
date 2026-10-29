"""Embed every observation with bge-m3 on an Apple Silicon GPU (MLX, fp16), for import-vectors.ts.

    python embed_store_mlx.py run      --out DIR [--db PATH]   embed each shard that has no file yet
    python embed_store_mlx.py assemble --out DIR               join the shards into DIR/store.f32 and DIR/store.ids.json

The worker embeds on the CPU, about 3 notes a second with half the threads, so a store of 200k
notes takes the better part of a day. This script did 212k notes in 23 minutes on an M5 Max (154
notes/s), and its vectors match the worker's full-precision ONNX ones (cosine 1.0000 on 2,000 notes):
the [CLS] state of bge-m3, 512 tokens, unit length, over the same document text the worker builds
(title, subtitle, and narrative or else text, each trimmed, empty ones dropped, joined by newlines,
first 32,768 characters).

The database is opened read-only and note texts are never written out; only vectors and ids are.
`run` is safe to stop and restart: finished shards are kept, and the id list is fixed on the first
run (DIR/manifest.json); delete DIR to start over. LIMIT=<n> caps the notes, for a trial run.
"""
import argparse, json, os, sqlite3, sys, time
from urllib.request import pathname2url

import numpy as np

DIM = 1024
REPO = "mlx-community/bge-m3-mlx-fp16"
SHARD = 1024
BATCH = 16
MAX_TOKENS = 512
MAX_CHARS = 32 * 1024


def doc_text(title, subtitle, narrative, text):
    """The worker's observationDocText: title, subtitle and narrative (or the legacy text), trimmed."""
    parts = [(p or "").strip() for p in (title, subtitle, narrative or text)]
    return "\n".join(p for p in parts if p)[:MAX_CHARS]


def plan_shards(ids, size):
    return [{"name": f"shard-{k:05d}", "ids": ids[i:i + size]} for k, i in enumerate(range(0, len(ids), size))]


def pending(shards, done):
    return [s for s in shards if s["name"] not in done]


def problems(v, n):
    """Why a shard's vectors are not fit to store: shape, non-finite values, length not 1."""
    if v.shape != (n, DIM):
        return [f"shape {v.shape}, expected {(n, DIM)}"]
    if not np.isfinite(v).all():
        return ["values not finite"]
    norms = np.linalg.norm(v, axis=1)
    if (norms < 0.99).any() or (norms > 1.01).any():
        return [f"norm outside 1 +- 0.01 (min {norms.min():.3f}, max {norms.max():.3f})"]
    return []


def assemble_shards(shards, arrays):
    parts = []
    for s in shards:
        a = arrays.get(s["name"])
        if a is None:
            raise ValueError(f"{s['name']} missing")
        if a.shape != (len(s["ids"]), DIM):
            raise ValueError(f"{s['name']} has shape {a.shape}, expected {(len(s['ids']), DIM)}")
        parts.append(a)
    ids = [i for s in shards for i in s["ids"]]
    return ids, (np.concatenate(parts) if parts else np.zeros((0, DIM), np.float32))


def connect(path):
    # Read-only needs a URI, where "?", "#" and "%" in the path must be escaped.
    return sqlite3.connect(f"file:{pathname2url(os.path.abspath(os.path.expanduser(path)))}?mode=ro", uri=True)


def manifest(out, db_path):
    path = os.path.join(out, "manifest.json")
    if os.path.exists(path):
        return json.load(open(path))
    ids = [r[0] for r in connect(db_path).execute(
        "SELECT id FROM observations WHERE COALESCE(TRIM(title),'')<>'' OR COALESCE(TRIM(subtitle),'')<>'' "
        "OR COALESCE(TRIM(narrative),'')<>'' OR COALESCE(TRIM(text),'')<>'' ORDER BY id")]
    limit = int(os.environ.get("LIMIT", "0"))
    if limit:
        ids = ids[:limit]
    m = {"repo": REPO, "shard": SHARD, "db": os.path.expanduser(db_path), "ids": ids, "created": time.strftime("%FT%T")}
    os.makedirs(out, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(m, f)
    os.replace(path + ".tmp", path)
    return m


def done_names(out):
    return {f[:-4] for f in os.listdir(out) if f.startswith("shard-") and f.endswith(".npy")}


def run(out, db_path):
    m = manifest(out, db_path)
    shards = plan_shards(m["ids"], m["shard"])
    todo = pending(shards, done_names(out))
    print(f"{len(m['ids'])} notes in {len(shards)} shards, {len(todo)} to do", flush=True)
    if not todo:
        return
    import mlx.core as mx
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from mlx_embeddings.utils import load_model
    from transformers import AutoTokenizer
    path = Path(snapshot_download(m["repo"]))
    tok = AutoTokenizer.from_pretrained(str(path))
    model = load_model(path)
    con = connect(m["db"])
    started, notes = time.time(), 0
    for k, s in enumerate(todo):
        docs = []
        for i in s["ids"]:
            row = con.execute("SELECT title, subtitle, narrative, text FROM observations WHERE id=?", (i,)).fetchone()
            if row is None:
                raise SystemExit(f"{s['name']}: note {i} no longer exists; delete {out} to start over")
            docs.append(doc_text(*row))
        enc = [tok(d, truncation=True, max_length=MAX_TOKENS)["input_ids"] for d in docs]
        order = np.argsort([len(e) for e in enc])  # similar lengths together: less padding
        vectors = np.zeros((len(docs), DIM), dtype=np.float32)
        for b in range(0, len(docs), BATCH):
            idx = order[b:b + BATCH]
            width = max(len(enc[j]) for j in idx)
            x = np.full((len(idx), width), tok.pad_token_id, dtype=np.int32)
            mask = np.zeros((len(idx), width), dtype=np.int32)
            for r, j in enumerate(idx):
                x[r, :len(enc[j])] = enc[j]
                mask[r, :len(enc[j])] = 1
            h = model(mx.array(x), attention_mask=mx.array(mask)).last_hidden_state
            cls = h[:, 0].astype(mx.float32)
            cls = cls / mx.linalg.norm(cls, axis=1, keepdims=True)
            mx.eval(cls)
            vectors[idx] = np.array(cls)
        bad = problems(vectors, len(docs))
        if bad:
            raise SystemExit(f"{s['name']}: {'; '.join(bad)}")
        tmp = os.path.join(out, s["name"] + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, vectors)
        os.replace(tmp, os.path.join(out, s["name"] + ".npy"))
        notes += len(docs)
        if k % 10 == 0 or k == len(todo) - 1:
            rate = notes / (time.time() - started)
            left = sum(len(x["ids"]) for x in todo[k + 1:])
            print(f"{time.strftime('%T')} {s['name']}: {notes} notes this run, {rate:.1f}/s, {left} left, "
                  f"about {left / rate / 60:.0f} min", flush=True)
    print("RUN COMPLETE", flush=True)


def assemble(out):
    m = json.load(open(os.path.join(out, "manifest.json")))
    shards = plan_shards(m["ids"], m["shard"])
    arrays = {s["name"]: np.load(os.path.join(out, s["name"] + ".npy"))
              for s in shards if os.path.exists(os.path.join(out, s["name"] + ".npy"))}
    ids, v = assemble_shards(shards, arrays)
    v.astype("<f4").tofile(os.path.join(out, "store.f32"))
    with open(os.path.join(out, "store.ids.json"), "w") as f:
        json.dump(ids, f)
    print(f"ASSEMBLED {len(ids)} vectors into {out}/store.f32 and store.ids.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["run", "assemble"])
    parser.add_argument("--out", required=True, help="folder for shards and the assembled files")
    parser.add_argument("--db", default="~/.claude-mem/claude-mem.db", help="claude-mem database (opened read-only)")
    a = parser.parse_args()
    run(a.out, a.db) if a.command == "run" else assemble(a.out)

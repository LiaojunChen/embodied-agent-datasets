"""Estimate token counts for every catalog entry, without downloading whole files.

Per data file:
  1. Sample ~400 records spread across the file
     - parquet: 4 row groups (image / binary columns are never read; image counts come from parquet metadata)
     - json / jsonl: HTTP range reads at 8 offsets; complete records are decoded from each window
  2. Text tokens: Qwen3.8 tokenizer over the trainable fields of a record (text_of: conversation + tool
     schema / task / answer; metadata, ids, urls, scores and continuous action/state arrays excluded;
     floats rendered with 4 significant digits)
  3. Visual items: max(<image>/<video> placeholders, image/video path fields + embedded image columns)
  4. Extrapolate: tokens_file = mean_per_record x records_in_file
     records_in_file: parquet metadata (exact), small json (exact), big json (file bytes / mean in-file record bytes)
Files beyond MAX_FILES per entry are extrapolated by bytes from the measured ones.

Usage: .venv/bin/python tokens.py [entry-key ...]   -> cache/tokens.json
"""
import concurrent.futures as cf
import json
import pathlib
import random
import re
import sys
import time
import urllib.parse
import urllib.request

from huggingface_hub import HfFileSystem
from tokenizers import Tokenizer

import catalog

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "cache" / "tokens.json"
TOKENIZER = "/mnt/vepfs01/output/watson.chen/phyRSI/Qwen3-27B/weights/tokenizer.json"

SAMPLE_RECORDS = 400
MAX_FILES = 12
WINDOWS = 16
WINDOW = 1 << 20
IMG_PATH = re.compile(r"\.(png|jpe?g|webp|bmp|gif|tiff?)$", re.I)
VID_PATH = re.compile(r"\.(mp4|avi|mov|mkv|webm)$", re.I)
PH_IMG = re.compile(r"<image>|<\|image_pad\|>|<img>")
PH_VID = re.compile(r"<video>|<\|video_pad\|>")
B64 = re.compile(r"^[A-Za-z0-9+/=\s]+$")

fs = HfFileSystem()
tok = Tokenizer.from_file(TOKENIZER)
DEC = json.JSONDecoder()


# ------------------------------------------------------------------ measuring
META_KEYS = {"metadata", "meta", "meta_data", "info", "extra_info"}


def walk(v, texts, vis, seen=None):
    """Collect string values; count distinct image / video references (metadata copies of a path are skipped)."""
    if seen is None:
        seen = vis.setdefault("_seen", set())
    if isinstance(v, str):
        if len(v) < 400 and "\n" not in v:
            if IMG_PATH.search(v):
                if v not in seen:
                    seen.add(v)
                    vis["img_paths"] += 1
                return
            if VID_PATH.search(v):
                if v not in seen:
                    seen.add(v)
                    vis["vid_paths"] += 1
                return
        if len(v) > 5000 and " " not in v[:5000] and B64.match(v[:5000]):
            vis["img_cols"] += 1  # base64-embedded image
            return
        vis["img_ph"] += len(PH_IMG.findall(v))
        vis["vid_ph"] += len(PH_VID.findall(v))
        texts.append(v)
    elif isinstance(v, dict):
        for k, x in v.items():
            kl = str(k).lower()
            if "depth" in kl:
                continue  # depth maps are auxiliary inputs, not RGB images or text
            if kl in META_KEYS:
                walk(x, texts, dict(img_paths=0, vid_paths=0, img_ph=0, vid_ph=0, img_cols=0, _seen=set()), set())
                continue  # metadata text still counts as text, its paths do not count as visual inputs
            walk(x, texts, vis, seen)
    elif isinstance(v, list):
        for x in v:
            walk(x, texts, vis, seen)
    elif isinstance(v, (bytes, bytearray)):
        vis["img_cols"] += 1
    elif v is not None and not isinstance(v, bool):
        texts.append(str(v))


# Trainable text: if a record has a conversation-like field, count that field plus a few companion
# fields (tool schemas, system prompt, task text, answer); otherwise count every field except DROP.
# Metadata, ids, urls, hashes, quality scores and continuous action/state arrays never reach the model.
CONV_KEYS = ("conversations", "conversation", "messages", "trajectory", "texts", "traj", "content", "prompt", "chosen", "rejected")
COMPANION = ("tools", "available_tools", "system", "instruction", "task", "goal", "answer", "answers", "golden_answers",
             "correct_answer", "response", "reward_model")
DROP = META_KEYS | {
    "id", "uuid", "uid", "qid", "key", "__key__", "__url__", "url", "image_url", "image_sha256", "sha256", "hash",
    "task_id", "trajectory_id", "episode_id", "video_uid", "node_id", "parent_id", "instance_id", "test_id", "run_id",
    "trial_name", "date", "created_at", "license", "source", "data_source", "model", "model_name", "model_provider", "agent",
    "split", "width", "height", "fps", "view", "question_quality_assessment", "response_quality_assessment", "filter_reason",
    "processing_info", "match_contexts", "matched_categories", "used_in", "subset_name", "target_tools", "action", "state",
    "action_is_pad", "program", "clip_similarity_vitb32", "clip_similarity_vitl14", "file_path", "img_filename", "image_path",
    "video_path", "ability", "eval_logs", "generated_patch", "model_patch", "exit_status", "finish_reason", "reward", "image_id",
    "base_commit", "image_name", "enable_thinking", "episode", "wp", "index", "step_id", "sample_id", "domain", "scenario",
    "parent", "screenshot_height", "screenshot_width", "annotation", "image_correspondence_ratings", "image_correspondence_min",
    "visual_dependency_ratings", "visual_dependency_min", "formatting_ratings", "formatting_min", "relevance_ratings",
    "relevance_min", "selected_step", "two_fps_timestamps", "raw_frames", "raw_timestamps", "annotator_unsure", "category",
    "video_duration", "video_source", "clip_start", "clip_end", "__index_level_0__", "trace_uid", "source_json_file",
    "source_trace_id", "source_trace_index", "source_step_id", "source_step_index", "repo", "dataset", "temperature",
    "chat_template_kwargs", "loss", "end_turn", "recipient", "weight",
    # role labels are replaced by chat-template special tokens at training time
    "role", "from", "class_", "type", "name_role", "speaker"}


def _jsonish(v):
    if isinstance(v, str) and v[:1] in "[{" and len(v) > 1:
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def collect(v, out):
    if isinstance(v, str):
        if len(v) < 400 and "\n" not in v and (IMG_PATH.search(v) or VID_PATH.search(v)):
            return  # media reference, not text
        if len(v) > 5000 and " " not in v[:5000] and B64.match(v[:5000]):
            return
        out.append(v)
    elif isinstance(v, dict):
        for k, x in v.items():
            kl = str(k).lower()
            if kl in DROP or "depth" in kl:
                continue
            collect(x, out)
    elif isinstance(v, list):
        for x in v:
            collect(x, out)
    elif isinstance(v, bool) or v is None or isinstance(v, (bytes, bytearray)):
        return
    elif isinstance(v, float):
        out.append(f"{v:.4g}")
    elif isinstance(v, int):
        out.append(str(v))


def text_of(r):
    if not isinstance(r, dict):
        out = []
        collect(r, out)
        return out
    conv = [(k, _jsonish(r[k])) for k in CONV_KEYS if k in r and r[k]]
    conv = [(k, v) for k, v in conv if isinstance(v, (list, dict))]
    out = []
    if conv:
        for _, v in conv:
            collect(v, out)
        for k in COMPANION:
            if r.get(k):
                collect(_jsonish(r[k]), out)
    else:
        collect(r, out)
    return out


def measure(records, extra_img_per_row=0.0):
    if not records:
        return None
    docs, imgs, vids = [], 0, 0
    for r in records:
        vis = dict(img_paths=0, vid_paths=0, img_ph=0, vid_ph=0, img_cols=0)
        walk(r, [], vis)  # visual items: whole record
        docs.append("\n".join(text_of(r)))  # text: trainable fields only
        imgs += max(vis["img_ph"], vis["img_paths"] + vis["img_cols"])
        vids += max(vis["vid_ph"], vis["vid_paths"])
    n = len(records)
    enc = tok.encode_batch(docs)
    img_per = imgs / n
    if extra_img_per_row:  # images held in parquet image columns (not read)
        img_per = max(img_per, extra_img_per_row)
    return dict(text=sum(len(e.ids) for e in enc) / n, images=img_per, videos=vids / n, n=n)


# -------------------------------------------------------------------- parquet
def is_blob(t):
    s = str(t)
    return s in ("binary", "large_binary") or "bytes: binary" in s or "bytes: large_binary" in s


def sample_parquet(repo, fpath, size, n_rg=8):
    import pyarrow.parquet as pq
    with fs.open(f"datasets/{repo}/{fpath}", "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        blobs = [fl.name for fl in pf.schema_arrow if is_blob(fl.type)]
        cols = [fl.name for fl in pf.schema_arrow if fl.name not in blobs]
        # visual items per row from footer metadata: value counts of the blob leaves
        img_vals = vid_vals = 0
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            for c in range(rg.num_columns):
                col = rg.column(c)
                top = col.path_in_schema.split(".")[0]
                if top in blobs and (col.path_in_schema.endswith(".bytes") or col.path_in_schema == top):
                    if re.search(r"mp4|video|vid", top, re.I):
                        vid_vals += col.num_values
                    else:
                        img_vals += col.num_values
        rows_total = max(1, md.num_rows)
        picks = sorted(set(int(i * md.num_row_groups / n_rg) for i in range(n_rg)))
        rows = []
        per = SAMPLE_RECORDS // len(picks) + 1
        for g in picks:
            if not cols:
                break
            try:
                tb = pf.read_row_group(g, columns=cols)
            except Exception:
                # huge nested cells ("Nested data conversions not implemented for chunked array outputs"):
                # stream small batches of the same row group instead
                part = []
                for b in pf.iter_batches(batch_size=16, row_groups=[g], columns=cols):
                    part += b.to_pylist()
                    if len(part) >= per:
                        break
                rows += part[:per]
                continue
            if tb.num_rows == 0:
                continue
            k = min(tb.num_rows, per)
            idx = sorted(random.Random(g).sample(range(tb.num_rows), k)) if tb.num_rows > k else list(range(tb.num_rows))
            rows += tb.take(idx).to_pylist()
    m = measure(rows, extra_img_per_row=img_vals / rows_total)
    if m and vid_vals:
        m["videos"] = max(m["videos"], vid_vals / rows_total)
    return m, md.num_rows, "exact"


# ----------------------------------------------------------------------- json
def http_range(repo, fpath, start, length):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(fpath)}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def similar(keys, ref):
    a, b = set(keys), set(ref)
    return bool(a) and len(a & b) / len(a | b) >= 0.5


def scan(text, ref=None, start=None):
    """Decode complete top-level-looking records from a text window; returns [(obj, in_file_bytes)].

    start: optional regex whose group(1) begins a record (for files that are one big nested dict).
    """
    out, covered = [], -1
    pat = start or r"(?:^|[\[,\n])\s*(\{)"
    for m in re.finditer(pat, text):
        i = m.start(1)
        if i < covered:
            continue
        try:
            obj, end = DEC.raw_decode(text, i)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if ref is not None and not similar(obj.keys(), ref):
            continue
        out.append((obj, len(text[i:end].encode("utf-8"))))
        covered = end
    return out


def sample_json(repo, fpath, size, start=None):
    if size <= 2 * WINDOW and start is None:
        text = http_range(repo, fpath, 0, size).decode("utf-8", "ignore")
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
            if fpath.endswith(".jsonl"):
                obj = [json.loads(ln) for ln in text.splitlines() if ln.strip().startswith("{")]
        if isinstance(obj, dict):
            vals = list(obj.values())
            if len(vals) == 1 and isinstance(vals[0], list):
                obj = vals[0]
            elif vals and all(isinstance(v, dict) for v in vals[:50]):
                obj = vals
        if isinstance(obj, list) and obj:
            recs = [r for r in obj if isinstance(r, dict)]
            pick = recs if len(recs) <= SAMPLE_RECORDS else random.Random(0).sample(recs, SAMPLE_RECORDS)
            return measure(pick), len(recs), "exact"
        return None, 0, "none"
    # big file: reference schema from the head, then spread windows
    win = WINDOW
    head = http_range(repo, fpath, 0, win).decode("utf-8", "ignore")
    first = scan(head, start=start)
    while not first and win < (64 << 20):
        win *= 4
        head = http_range(repo, fpath, 0, win).decode("utf-8", "ignore")
        first = scan(head, start=start)
    if not first:
        return None, 0, "none"
    ref = first[0][0].keys()
    got = [x for x in first if similar(x[0].keys(), ref)]
    for q in range(1, WINDOWS):
        off = int(size * q / WINDOWS)
        chunk = http_range(repo, fpath, off, min(win, size - off)).decode("utf-8", "ignore")
        got += scan("\n" + chunk, ref, start)[1:]  # first hit may be a truncated tail
    recs = [g[0] for g in got]
    per_bytes = sum(g[1] for g in got) / len(got)
    pick = recs if len(recs) <= SAMPLE_RECORDS else random.Random(0).sample(recs, SAMPLE_RECORDS)
    return measure(pick), int(size / (per_bytes + 2)), "bytes"


# --------------------------------------------------------------------- driver
def estimate_file(repo, fpath, size, start=None, n_rg=8):
    t0 = time.time()
    if fpath.lower().endswith(".parquet"):
        m, n, how = sample_parquet(repo, fpath, size, n_rg)
    else:
        m, n, how = sample_json(repo, fpath, size, start)
    if not m:
        return dict(repo=repo, path=fpath, size=size, error="no records decoded")
    return dict(repo=repo, path=fpath, size=size, records=n, how=how, sampled=m["n"], text_per=m["text"],
                img_per=m["images"], vid_per=m["videos"], text_tokens=int(m["text"] * n),
                images=int(round(m["images"] * n)), videos=int(round(m["videos"] * n)), secs=round(time.time() - t0, 1))


def plan_files(entry, trees):
    files = []
    for repo, pats in entry["files"].items():
        tree = trees.get(repo)
        if not isinstance(tree, list):
            continue
        for f in tree:
            fp = f["path"]
            if fp.lower().endswith((".json", ".jsonl", ".parquet")) and any(re.fullmatch(p, fp) for p in pats):
                files.append(dict(repo=repo, path=fp, size=f["size"]))
    return files


def estimate_entry(key, entry, trees):
    files = plan_files(entry, trees)
    if not files:
        return dict(key=key, error="no data files matched", files=[])
    files.sort(key=lambda f: -f["size"])
    if len(files) <= MAX_FILES:
        chosen = files
    else:
        chosen = files[:MAX_FILES // 2] + random.Random(0).sample(files[MAX_FILES // 2:], MAX_FILES // 2)
    res = []
    for f in chosen:
        try:
            res.append(estimate_file(f["repo"], f["path"], f["size"], entry.get("json_start"), entry.get("rgs", 8)))
        except Exception as e:
            res.append(dict(repo=f["repo"], path=f["path"], size=f["size"], error=f"{type(e).__name__}: {str(e)[:160]}"))
    ok = [r for r in res if "error" not in r]
    if not ok:
        return dict(key=key, error="; ".join(sorted({r["error"] for r in res}))[:300], files=res)
    done = {(r["repo"], r["path"]) for r in ok}
    rest = [f for f in files if (f["repo"], f["path"]) not in done]
    scale = 1 + sum(f["size"] for f in rest) / max(1, sum(r["size"] for r in ok))
    agg = lambda k: int(sum(r[k] for r in ok) * scale)
    return dict(key=key, n_files=len(files), n_measured=len(ok), bytes=sum(f["size"] for f in files),
                records=agg("records"), text_tokens=agg("text_tokens"), images=agg("images"), videos=agg("videos"),
                exact_records=all(r["how"] == "exact" for r in ok) and not rest, files=res)


def hf_sizes(repo):
    """Exact row counts per (config, split) from the datasets-server /size API, or {}."""
    try:
        with urllib.request.urlopen("https://datasets-server.huggingface.co/size?dataset=" + urllib.parse.quote(repo), timeout=120) as r:
            d = json.load(r)
        return {(s["config"], s["split"]): (s.get("estimated_num_rows") or s["num_rows"]) for s in d["size"]["splits"]}
    except Exception:
        return {}


def estimate_stratified(key, entry, trees):
    """Huge multi-config parquet repos: measure 1-2 files per config, scale by exact /size row counts.

    entry["strata"] = {repo: dict(group=regex whose group(1) maps a file path to its config name,
                                  splits=("train",))}
    Byte-scaling is wrong for these repos (image bytes dominate file size), so rows come from /size;
    a config missing from /size falls back to per-file parquet row counts x file count.
    """
    res, strata = [], []
    for repo, spec in entry["strata"].items():
        sizes = hf_sizes(repo)
        rows_by_cfg = {}
        for (cfg, split), n in sizes.items():
            if spec.get("key") == "split":  # repo names its parts as splits of one config
                if cfg == spec["cfg"]:
                    rows_by_cfg[split] = rows_by_cfg.get(split, 0) + n
            elif split in spec.get("splits", ("train",)):
                rows_by_cfg[cfg] = rows_by_cfg.get(cfg, 0) + n
        if spec.get("rows_cfg"):  # a single file group maps to one named config
            rows_by_cfg = {g: rows_by_cfg.get(spec["rows_cfg"]) for g in ("data",) if rows_by_cfg.get(spec["rows_cfg"])}
        by_cfg = {}
        for f in plan_files(dict(files={repo: entry["files"][repo]}), trees):
            m = re.match(spec["group"], f["path"])
            if m:
                by_cfg.setdefault(m.group(1), []).append(f)
        def measure_cfg(item):
            cfg, fl = item
            fl.sort(key=lambda f: f["path"])
            k = spec.get("picks", 1 if spec.get("one") else 2)
            if k <= 2:
                picks = [fl[len(fl) // 2]] + ([fl[0]] if len(fl) > 4 and k == 2 else [])
            else:  # evenly spaced files for high-variance repos
                picks = [fl[int(i * len(fl) / k)] for i in range(k)]
            out = []
            for f in picks:
                try:
                    out.append(estimate_file(f["repo"], f["path"], f["size"], n_rg=spec.get("rgs", 8)))
                except Exception as e:
                    out.append(dict(repo=repo, path=f["path"], size=f["size"], error=f"{type(e).__name__}: {str(e)[:160]}"))
            return cfg, fl, out

        with cf.ThreadPoolExecutor(spec.get("workers", 6)) as pool:
            measured = list(pool.map(measure_cfg, sorted(by_cfg.items())))
        for cfg, fl, out in measured:
            res += out
            ms = [r for r in out if "error" not in r]
            if not ms:
                strata.append(dict(config=cfg, error="no file measured"))
                continue
            w = sum(m["sampled"] for m in ms)
            mean = lambda k: sum(m[k] * m["sampled"] for m in ms) / w
            n_rows, how = rows_by_cfg.get(cfg), "size-api"
            if n_rows is None:
                n_rows, how = int(sum(m["records"] for m in ms) / len(ms) * len(fl)), "rows-per-file"
            strata.append(dict(config=cfg, rows=int(n_rows), how=how, files=len(fl), text_per=mean("text_per"),
                               img_per=mean("img_per"), vid_per=mean("vid_per")))
    ok = [s for s in strata if "error" not in s]
    if not ok:
        return dict(key=key, error="no stratum measured", files=res)
    return dict(key=key, n_files=sum(s["files"] for s in ok), n_measured=sum(1 for r in res if "error" not in r),
                bytes=0, records=int(sum(s["rows"] for s in ok)), text_tokens=int(sum(s["rows"] * s["text_per"] for s in ok)),
                images=int(sum(s["rows"] * s["img_per"] for s in ok)), videos=int(sum(s["rows"] * s["vid_per"] for s in ok)),
                exact_records=all(s["how"] == "size-api" for s in ok), strata=strata, files=res)


def save(key, result):
    """Merge one result into cache/tokens.json under an exclusive lock (safe with concurrent runs)."""
    import fcntl
    with open(OUT.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = json.loads(OUT.read_text()) if OUT.exists() else {}
        cur[key] = result
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1))
        tmp.replace(OUT)


def main(only):
    trees = json.loads((ROOT / "cache" / "trees.json").read_text())
    todo = [(k, e) for k, e in catalog.ENTRIES.items() if e.get("files") and (not only or k in only)]
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(estimate_stratified if e.get("strata") else estimate_entry, k, e, trees): k for k, e in todo}
        for fu in cf.as_completed(futs):
            k = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                r = dict(key=k, error=f"{type(e).__name__}: {str(e)[:200]}")
            save(k, r)
            if "error" in r:
                print(f"FAIL {k}: {r['error']}", flush=True)
            else:
                print(f"OK   {k:<28} rec={r['records']:>11,} text={r['text_tokens'] / 1e9:8.3f}B img={r['images']:>11,} "
                      f"vid={r['videos']:>9,} files={r['n_measured']}/{r['n_files']}", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]))

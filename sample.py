"""Pull one real record per dataset into samples.json (+ thumbnails under site/samples/).

Run locally when the dataset list changes: python3 sample.py [name ...]
CI does not run this (samples are committed), so a flaky upstream never breaks the page.

Sources, in order of preference:
  viewer       HF datasets-server first-rows API
  file         HTTP range read of the head of a raw JSON / JSONL file in the repo
  illustrative hand-written format illustration (gated repos, paper-only entries)
"""
import hashlib
import io
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request

import data
from illustrative import ILLUSTRATIVE

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "samples.json"
IMG_DIR = ROOT / "site" / "samples"
DS = "https://datasets-server.huggingface.co"

# Per-dataset overrides. Default: first HF id, first config/split, row 0.
OVERRIDES = {
    "RoboFAC": dict(file="MINT-SJTU/RoboFAC-dataset/training_qa.json", bytes=12_000_000,
                    pick=lambda r: "what led to" in json.dumps(r) or "explanation of why" in json.dumps(r)),
    "EmbodiedBench 轨迹": dict(file="EmbodiedBench/EB-Man_trajectory_dataset/eb-man_dataset_multi_step.json", bytes=3_000_000),
    "VSI-590K (Cambrian-S)": dict(file="nyu-visionx/VSI-590K/vsi_590k.jsonl", bytes=20_000),
    "RoboPoint": dict(file="wentao-yuan/robopoint-data/robopoint_1432k.json", bytes=200_000),
    "OS-Atlas": dict(file="OS-Copilot/OS-Atlas-data/mobile_domain/aw_mobile.json", bytes=200_000),
    "ADP (Agent Data Protocol)": dict(config="agenttuning_alfworld", split="std"),
    "AgentGym": dict(hf="AgentGym/AgentTraj-L"),
}

MAX_STR, MAX_KEYS, MAX_IMAGES = 700, 16, 3
HEAD, TAIL = 3, 3  # long lists (trajectories) keep the opening and the outcome


def get_json(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def head_records(path, nbytes):
    """Decode as many complete top-level records as fit in the first nbytes of a JSON array / JSONL file."""
    repo, fname = "/".join(path.split("/")[:2]), "/".join(path.split("/")[2:])
    req = urllib.request.Request(f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}",
                                 headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", errors="ignore")
    dec, i, out = json.JSONDecoder(), 0, []
    text = text.lstrip()
    if text.startswith("["):
        i = 1
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        out.append(obj)
    return out


def thumb(url, slug, n):
    from PIL import Image
    with urllib.request.urlopen(url, timeout=60) as r:
        im = Image.open(io.BytesIO(r.read())).convert("RGB")
    im.thumbnail((420, 420))
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{slug}_{n}.jpg"
    im.save(IMG_DIR / name, quality=82)
    return f"samples/{name}"


def shrink(v, ctx, depth=0):
    """Truncate a record for display; download viewer images as local thumbnails."""
    if isinstance(v, dict) and "src" in v and str(v.get("src", "")).startswith("http"):
        if len(ctx["images"]) < MAX_IMAGES:
            try:
                ctx["images"].append(thumb(v["src"], ctx["slug"], len(ctx["images"])))
                return f"<image #{len(ctx['images'])}>"
            except Exception as e:
                print(f"  image fail: {e}", file=sys.stderr)
        return "<image>"
    if depth > 7:
        return "…"
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{" and len(s) > 2:  # JSON stored as a string column
            try:
                return shrink(json.loads(s), ctx, depth)
            except ValueError:
                pass
        return v if len(v) <= MAX_STR else v[:MAX_STR] + f"…（共 {len(v):,} 字符，已截断）"
    if isinstance(v, list):
        if len(v) <= HEAD + TAIL + 1:
            return [shrink(x, ctx, depth + 1) for x in v]
        return ([shrink(x, ctx, depth + 1) for x in v[:HEAD]] + [f"…（共 {len(v)} 项，省略中间 {len(v) - HEAD - TAIL} 项）"]
                + [shrink(x, ctx, depth + 1) for x in v[-TAIL:]])
    if isinstance(v, dict):
        keys = list(v)[:MAX_KEYS]
        out = {k: shrink(v[k], ctx, depth + 1) for k in keys}
        if len(v) > MAX_KEYS:
            out["…"] = f"另有 {len(v) - MAX_KEYS} 个字段"
        return out
    return v


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "ds"


def sample_one(d):
    name, ov = d["name"], OVERRIDES.get(d["name"], {})
    ctx = dict(slug=slugify(name) + "-" + hashlib.md5(name.encode()).hexdigest()[:6], images=[])
    for old in IMG_DIR.glob(ctx["slug"] + "_*.jpg"):
        old.unlink()
    if name in ILLUSTRATIVE:
        ill = ILLUSTRATIVE[name]
        return dict(kind="illustrative", source=ill["why"], url=None, record=ill["record"], images=[])
    if "file" in ov:
        recs = head_records(ov["file"], ov["bytes"])
        pick = ov.get("pick")
        idx = next((i for i, r in enumerate(recs) if pick(r)), 0) if pick else 0
        repo = "/".join(ov["file"].split("/")[:2])
        return dict(kind="file", source=f"{ov['file']} · 第 {idx + 1} 条",
                    url=f"https://huggingface.co/datasets/{repo}", record=shrink(recs[idx], ctx), images=ctx["images"])
    hid = ov.get("hf") or d["hf"][0]
    splits = get_json(f"{DS}/splits?dataset={urllib.parse.quote(hid)}")["splits"]
    s = next((x for x in splits if x["config"] == ov.get("config", x["config"]) and x["split"] == ov.get("split", x["split"])), splits[0])
    fr = get_json(f"{DS}/first-rows?" + urllib.parse.urlencode(dict(dataset=hid, config=s["config"], split=s["split"])))
    row = fr["rows"][0]["row"]
    if fr["rows"][0].get("truncated_cells"):  # first-rows clips big cells; /rows returns them whole
        q = urllib.parse.urlencode(dict(dataset=hid, config=s["config"], split=s["split"], offset=0, length=1))
        row = get_json(f"{DS}/rows?{q}", timeout=120)["rows"][0]["row"]
    return dict(kind="viewer", source=f"{hid} · {s['config']}/{s['split']} · row 0",
                url=f"https://huggingface.co/datasets/{hid}/viewer/{s['config']}/{s['split']}?row=0",
                record=shrink(row, ctx), images=ctx["images"])


def main(only):
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for d in data.DATASETS:
        if only and d["name"] not in only:
            continue
        try:
            out[d["name"]] = sample_one(d)
            s = out[d["name"]]
            print(f"OK   {s['kind']:<12} {d['name']}  imgs={len(s['images'])}")
        except Exception as e:  # keep the previous sample if there is one
            print(f"FAIL {d['name']}: {e}", file=sys.stderr)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    missing = [d["name"] for d in data.DATASETS if d["name"] not in out]
    print(f"{len(out)} samples written; missing: {missing}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))

"""Token-budgeted mixture versions over catalog.py + cache/tokens.json.

Accounting per entry:
  text   = Qwen3.8 tokens over all string values (tokens.py)
  vision = images x vtok.image + videos x vtok.video   (per-entry budgets in catalog.py)
  total  = text + vision
Only tier == "core" entries are mixed (dup / ref / rl / eval tiers are listed on the page but not counted).

Filling a direction's budget, "large single-source first":
  1. entries sorted by (single-source first, priority, larger first)
  2. the first single-source entry may take at most CAP_FIRST of the budget, every other entry at most CAP_OTHER,
     so a 1B direction is never one dataset (keeps the version cleanable but not monocultural)
  3. a second pass relaxes the caps if the budget is still not met
  4. never repeat data: if a direction runs out, the shortfall is reported, not padded
"""
import json
import pathlib

import catalog

ROOT = pathlib.Path(__file__).parent
CAP_FIRST, CAP_OTHER = 0.60, 0.35

VERSIONS = [
    dict(id="v05b", name="每方向 0.5B（3.5B）", per_dir={d["id"]: 0.5e9 for d in catalog.DIRECTIONS},
         desc="七个方向各 0.5B：最小可跑通版本，用于清洗流程与训练脚本联调"),
    dict(id="v1b", name="每方向 1B（7B）", per_dir={d["id"]: 1e9 for d in catalog.DIRECTIONS},
         desc="七个方向各 1B token。用于快速验证方向配比、做方向消融的基础版本"),
    dict(id="v3b", name="每方向 3B（21B）", per_dir={d["id"]: 3e9 for d in catalog.DIRECTIONS},
         desc="七个方向各 3B。数据不足的方向如实标出缺口，不重复采样补齐"),
    dict(id="v5b", name="每方向 5B（35B）", per_dir={d["id"]: 5e9 for d in catalog.DIRECTIONS},
         desc="七个方向各 5B，接近单方向大源的规模上限"),
    dict(id="emb10", name="具身侧重 10B", per_dir=dict(A=2e9, B=2e9, C=1e9, D=1.5e9, E=2e9, F=1e9, G=0.5e9),
         desc="A/B/E 各 2B、D 1.5B、C/F 各 1B、G 0.5B：具身感知与决策为主，保留少量通用与 agentic"),
    dict(id="full", name="全量上限", per_dir=None,
         desc="每个方向用尽全部可计数据（已去重），给出各方向能达到的上限"),
]


def entry_tokens(key, e, tok, overrides):
    t = tok.get(key)
    est = False
    if not t or "error" in t:
        if key not in overrides:
            return None
        t, est = overrides[key], True
    imgs = t.get("images", 0) if e.get("count_images", True) else 0
    vids = t.get("videos", 0)
    pr = e.get("per_row") or {}
    if "image" in pr:
        imgs = int(t.get("records", 0) * pr["image"])
    if "video" in pr:
        vids = int(t.get("records", 0) * pr["video"])
    vis = imgs * e["vtok"].get("image", 0) + vids * e["vtok"].get("video", 0)
    return dict(text=int(t["text_tokens"]), vis=int(vis), total=int(t["text_tokens"] + vis), images=int(imgs),
                videos=int(vids), records=int(t.get("records", 0)), estimated=est)


def pool_by_dir(tok, overrides):
    pool = {}
    for key, e in catalog.ENTRIES.items():
        if e["tier"] != "core":
            continue
        n = entry_tokens(key, e, tok, overrides)
        if n is None or n["total"] <= 0:
            continue
        pool.setdefault(e["dir"], []).append((key, e, n))
    for d in pool:
        pool[d].sort(key=lambda x: (0 if x[1]["single"] else 1, x[1].get("priority", 5), -x[2]["total"]))
    return pool


def fill(items, budget):
    take = {k: 0 for k, _, _ in items}
    left = budget
    for caps in ((CAP_FIRST, CAP_OTHER), (1.0, 1.0)):
        for i, (k, e, n) in enumerate(items):
            if left <= 0:
                break
            cap = (caps[0] if i == 0 and e["single"] else caps[1]) * budget
            room = min(n["total"] - take[k], cap - take[k], left)
            if room > 0:
                take[k] += room
                left -= room
    return take, max(left, 0)


def build(tok, overrides=None):
    pool = pool_by_dir(tok, overrides or {})
    out = []
    for v in VERSIONS:
        dirs = []
        for d in catalog.DIRECTIONS:
            items = pool.get(d["id"], [])
            avail = sum(n["total"] for _, _, n in items)
            budget = avail if v["per_dir"] is None else v["per_dir"][d["id"]]
            take, short = fill(items, budget) if v["per_dir"] is not None else ({k: n["total"] for k, _, n in items}, 0)
            picks = []
            for k, e, n in items:
                t = take[k]
                if t <= 0:
                    continue
                f = t / n["total"]
                picks.append(dict(key=k, name=e["name"], single=e["single"], take=int(t), frac=f, avail=n["total"],
                                  text=int(n["text"] * f), vis=int(n["vis"] * f), records=int(n["records"] * f),
                                  images=int(n["images"] * f), videos=int(n["videos"] * f), estimated=n["estimated"]))
            got = sum(p["take"] for p in picks)
            dirs.append(dict(dir=d["id"], budget=int(budget), got=int(got), short=int(short), avail=int(avail), picks=picks))
        out.append(dict(id=v["id"], name=v["name"], desc=v["desc"], dirs=dirs,
                        total=sum(x["got"] for x in dirs), budget=sum(x["budget"] for x in dirs)))
    return out


def summary(tok, overrides=None):
    """Per-entry token table for the page (all tiers, so dup / ref entries still show their size)."""
    rows = {}
    for key, e in catalog.ENTRIES.items():
        n = entry_tokens(key, e, tok, overrides or {})
        if n:
            rows[key] = n
    return rows


if __name__ == "__main__":
    tok = json.loads((ROOT / "tokens.json").read_text())
    ov = json.loads((ROOT / "overrides.json").read_text()) if (ROOT / "overrides.json").exists() else {}
    for v in build(tok, ov):
        print(f"== {v['id']} {v['name']}  {v['total'] / 1e9:.2f}B / {v['budget'] / 1e9:.2f}B")
        for d in v["dirs"]:
            s = ", ".join(f"{p['name']} {p['take'] / 1e9:.2f}B" for p in d["picks"])
            flag = f" SHORT {d['short'] / 1e9:.2f}B" if d["short"] else ""
            print(f"   {d['dir']} {d['got'] / 1e9:6.2f}/{d['budget'] / 1e9:5.2f}B (avail {d['avail'] / 1e9:.2f}){flag} :: {s}")

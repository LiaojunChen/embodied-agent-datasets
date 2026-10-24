"""Render site/index.html: datasets by direction (数据分类.png), token counts, token-budgeted mixture versions.

Inputs (all committed): catalog.py, tokens.json (from tokens.py + export_tokens.py), overrides.json, samples_v2.json, stats2.json
Outputs: site/v2.html, site/mixes/<version>.json (per-version manifest: entry, repo, files, share, tokens)

Usage: .venv/bin/python build2.py            # refresh HF downloads / likes, then render
       .venv/bin/python build2.py --offline  # render from cached stats2.json
"""
import concurrent.futures as cf
import datetime as dt
import json
import pathlib
import sys
import urllib.request

import catalog
import mix

ROOT = pathlib.Path(__file__).parent
STATS = ROOT / "stats2.json"
OUT = ROOT / "site" / "index.html"
API = "https://huggingface.co/api/datasets/{}?expand[]=downloads&expand[]=downloadsAllTime&expand[]=likes&expand[]=gated&expand[]=cardData"


def fetch_one(hf_id):
    try:
        with urllib.request.urlopen(API.format(hf_id), timeout=60) as r:
            d = json.load(r)
        lic = (d.get("cardData") or {}).get("license")
        return hf_id, dict(downloads=d.get("downloads", 0), downloadsAllTime=d.get("downloadsAllTime", 0),
                           likes=d.get("likes", 0), gated=bool(d.get("gated")),
                           license=lic if isinstance(lic, str) else (", ".join(lic) if lic else None))
    except Exception as e:
        print(f"WARN {hf_id}: {e}", file=sys.stderr)
        return hf_id, None


def fetch_stats():
    ids = sorted({i for e in catalog.ENTRIES.values() for i in e["hf"]})
    with cf.ThreadPoolExecutor(8) as ex:
        stats = dict(ex.map(fetch_one, ids))
    old = json.loads(STATS.read_text()) if STATS.exists() else {"stats": {}}
    for k, v in stats.items():
        if v is None:
            stats[k] = old["stats"].get(k)
    out = {"fetchedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "stats": stats}
    STATS.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def manifest(version, tok):
    """Per-version manifest: which files of which repo, and what share of each entry to take."""
    out = dict(version=version["id"], name=version["name"], desc=version["desc"], total_tokens=version["total"],
               token_definition="text = Qwen3.8 tokens over trainable fields; vision = images x vtok.image + videos x vtok.video",
               directions=[])
    for d in version["dirs"]:
        items = []
        for p in d["picks"]:
            e = catalog.ENTRIES[p["key"]]
            items.append(dict(entry=p["key"], name=p["name"], hf=e["hf"], share=round(p["frac"], 4), tokens=p["take"],
                              text_tokens=p["text"], vision_tokens=p["vis"], est_records=p["records"],
                              sampling="随机抽取该条目 {:.1%} 的样本（按样本数，不截断单条）".format(p["frac"]) if p["frac"] < 0.999 else "全量",
                              file_patterns=e["files"], n_files=(tok.get(p["key"]) or {}).get("n_files")))
        out["directions"].append(dict(dir=d["dir"], budget=d["budget"], tokens=d["got"], shortfall=d["short"], items=items))
    return out


def render(stats):
    tok = json.loads((ROOT / "tokens.json").read_text())
    ov = json.loads((ROOT / "overrides.json").read_text())
    samples = json.loads((ROOT / "samples_v2.json").read_text())
    counts = mix.summary(tok, ov)
    versions = mix.build(tok, ov)
    used = {}
    for v in versions:
        for d in v["dirs"]:
            for p in d["picks"]:
                used.setdefault(p["key"], {})[v["id"]] = p["take"]
    entries = []
    for key, e in catalog.ENTRIES.items():
        per = [dict(id=i, **(stats["stats"].get(i) or {})) for i in e["hf"]]
        known = [p for p in per if "likes" in p]
        agg = None
        if known:
            # entries split from one repo (ShareRobot / Cosmos / EO) show the repo's stats, not a sum over duplicates
            agg = dict(downloads=sum(p["downloads"] for p in known), downloadsAllTime=sum(p["downloadsAllTime"] for p in known),
                       likes=sum(p["likes"] for p in known), gated=any(p["gated"] for p in known),
                       license=next((p.get("license") for p in known if p.get("license")), None))
        n = counts.get(key)
        t = tok.get(key) or {}
        est = n.get("estimated") if n else False
        entries.append(dict(
            key=key, dir=e["dir"], also=e["also"], name=e["name"], hf=per, links=e["links"], single=e["single"], tier=e["tier"],
            new=bool(e.get("new")), group=e.get("group"), modality=e["modality"], role=e["role"], note=e["note"], agg=agg,
            vtok=e["vtok"], priority=e.get("priority", 5), domain=e.get("domain"),
            tokens=None if not n else dict(text=n["text"], vis=n["vis"], total=n["total"], images=n["images"], videos=n["videos"],
                                           records=n["records"], estimated=est,
                                           basis=ov.get(key, {}).get("basis") if est else None,
                                           exact_records=bool(t.get("exact_records")), files=t.get("n_files"), measured=t.get("n_measured")),
            used=used.get(key, {}), sample=samples.get(key)))
    vers = [dict(id=v["id"], name=v["name"], desc=v["desc"], total=v["total"], budget=v["budget"],
                 dirs=[dict(dir=d["dir"], budget=d["budget"], got=d["got"], short=d["short"], avail=d["avail"],
                            picks=[dict(key=p["key"], name=p["name"], take=p["take"], text=p["text"], vis=p["vis"], frac=p["frac"],
                                        single=p["single"], estimated=p["estimated"]) for p in d["picks"]]) for d in v["dirs"]])
            for v in versions]
    mdir = ROOT / "site" / "mixes"
    mdir.mkdir(parents=True, exist_ok=True)
    for v in versions:
        (mdir / f"{v['id']}.json").write_text(json.dumps(manifest(v, tok), ensure_ascii=False, indent=1))
    payload = dict(fetchedAt=stats["fetchedAt"], builtAt=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                   directions=catalog.DIRECTIONS, entries=entries, versions=vers, domains=catalog.DOMAIN_ZH,
                   vtok_rule="Qwen3.x ViT：16px patch、2×2 merge，每 32×32 像素 1 个 token；图像按原分辨率钳制到 64–1024 token；视频取 16 帧、每 2 帧一组、每帧上限 448×448（196 token）",
                   caps=dict(first=mix.CAP_FIRST, other=mix.CAP_OTHER))
    tpl = (ROOT / "template2.html").read_text()
    js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(tpl.replace("/*__DATA__*/null", js))
    print(f"wrote {OUT} ({len(entries)} entries, {len(vers)} versions, stats {stats['fetchedAt']})")


if __name__ == "__main__":
    stats = json.loads(STATS.read_text()) if "--offline" in sys.argv and STATS.exists() else fetch_stats()
    render(stats)

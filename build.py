"""Fetch HF stats for every dataset in data.py and render site/index.html.

Usage: python3 build.py            # fetch live stats, then render
       python3 build.py --offline  # render from cached stats.json
"""
import concurrent.futures as cf
import datetime as dt
import json
import pathlib
import sys
import urllib.request

import data

ROOT = pathlib.Path(__file__).parent
CACHE = ROOT / "stats.json"
OUT = ROOT / "site" / "index.html"
API = "https://huggingface.co/api/datasets/{}?expand[]=downloads&expand[]=downloadsAllTime&expand[]=likes&expand[]=gated&expand[]=lastModified"


def fetch_one(hf_id):
    try:
        with urllib.request.urlopen(API.format(hf_id), timeout=30) as r:
            d = json.load(r)
        return hf_id, dict(downloads=d.get("downloads", 0), downloadsAllTime=d.get("downloadsAllTime", 0),
                           likes=d.get("likes", 0), gated=bool(d.get("gated")), lastModified=d.get("lastModified", ""))
    except Exception as e:  # keep building; the page shows "—" for missing stats
        print(f"WARN {hf_id}: {e}", file=sys.stderr)
        return hf_id, None


def fetch_all():
    ids = sorted({i for d in data.DATASETS for i in d["hf"]})
    with cf.ThreadPoolExecutor(8) as ex:
        stats = dict(ex.map(fetch_one, ids))
    old = json.loads(CACHE.read_text()) if CACHE.exists() else {"stats": {}}
    for k, v in stats.items():  # fall back to last good value on transient errors
        if v is None:
            stats[k] = old["stats"].get(k)
    out = {"fetchedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "stats": stats}
    CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def render(cache):
    # samples.json is produced by sample.py and committed; thumbnails live in site/samples/
    samples = json.loads((ROOT / "samples.json").read_text()) if (ROOT / "samples.json").exists() else {}
    rows = []
    for d in data.DATASETS:
        per = [dict(id=i, **(cache["stats"].get(i) or {})) for i in d["hf"]]
        known = [p for p in per if "likes" in p]
        agg = None
        if known:
            agg = dict(downloads=sum(p["downloads"] for p in known), downloadsAllTime=sum(p["downloadsAllTime"] for p in known),
                       likes=sum(p["likes"] for p in known), gated=any(p["gated"] for p in known))
        rows.append(dict(cat=d["cat"], name=d["name"], hf=per, links=d["links"], scale=d["scale"], ability=d["ability"],
                         stages=d["stages"], note=d.get("note", ""), agg=agg, sample=samples.get(d["name"])))
    payload = dict(fetchedAt=cache["fetchedAt"], categories=data.CATEGORIES, datasets=rows,
                   stages=data.STAGES, evals=data.EVALS)
    tpl = (ROOT / "template.html").read_text()
    # </ inside JSON would close the script tag
    js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(tpl.replace("/*__DATA__*/null", js))
    print(f"wrote {OUT} ({len(rows)} datasets, stats {cache['fetchedAt']})")


if __name__ == "__main__":
    cache = json.loads(CACHE.read_text()) if "--offline" in sys.argv else fetch_all()
    render(cache)

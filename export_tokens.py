"""Slim cache/tokens.json (local, with per-file sampling detail) into tokens.json (committed, read by build2.py / CI)."""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
KEEP = ("records", "text_tokens", "images", "videos", "n_files", "n_measured", "bytes", "exact_records")
src = json.loads((ROOT / "cache" / "tokens.json").read_text())
out = {}
for k, v in sorted(src.items()):
    if "error" in v:
        continue
    out[k] = {x: v[x] for x in KEEP if x in v}
    if v.get("strata"):
        out[k]["strata"] = [dict(config=s["config"], rows=s["rows"], how=s["how"], text_per=round(s["text_per"], 1),
                                 img_per=round(s["img_per"], 3), vid_per=round(s["vid_per"], 3))
                            for s in v["strata"] if "error" not in s]
(ROOT / "tokens.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"tokens.json: {len(out)} entries")

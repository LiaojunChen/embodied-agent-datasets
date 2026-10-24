"""One real record per catalog entry -> samples_v2.json (+ thumbnails under site/samples/).

Run locally after catalog changes: .venv/bin/python sample2.py [entry-key ...]
Reuses samples.json records for entries that map 1:1 to the old dataset list; every split
or new entry gets its own record from the right config / file. Gated and paper-only entries
fall back to the labeled format illustrations in illustrative.py.
"""
import hashlib
import json
import pathlib
import sys
import urllib.parse

import catalog
from illustrative import ILLUSTRATIVE
from sample import DS, IMG_DIR, get_json, head_records, shrink, slugify

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "samples_v2.json"
OLD = json.loads((ROOT / "samples.json").read_text())

# entry -> name in samples.json when the old record is the right one for this entry
REUSE = {
    "pixmo-points": "PixMo-Points", "os-atlas": "OS-Atlas", "robopoint": "RoboPoint", "refspatial": "RefSpatial (RoboRefer)",
    "uground": "UGround", "vsi-590k": "VSI-590K (Cambrian-S)", "robovqa": "RoboVQA", "robofac": "RoboFAC", "aha": "AHA",
    "when2call": "When2Call", "hermes-fc": "Hermes Function Calling", "xlam": "xLAM-60k (APIGen)", "toolace": "ToolACE",
    "agentnet": "AgentNet (OpenCUA)", "androidcontrol": "AndroidControl", "gui-odyssey": "GUI-Odyssey",
    "embodied-reasoner": "Embodied-Reasoner", "ebench": "EmbodiedBench 轨迹", "era": "ERA 先验数据", "agenttraj-l": "AgentGym",
    "egoplan": "EgoPlan-IT / EgoPlan-Bench2", "toucan": "TOUCAN", "toucan-c": "TOUCAN", "nemotron-agentic-v2": "Nemotron-SFT-Agentic",
    "swe-zero": "NVIDIA SWE-Zero / SWE-Hero", "swe-smith-traj": "SWE-smith 轨迹", "nebius-swe-agent": "Nebius SWE-agent 轨迹",
    "r2e-gym": "R2E-Gym", "adp": "ADP (Agent Data Protocol)", "openresearcher": "OpenResearcher", "miroverse": "MiroVerse",
    "toolbench": "ToolBench", "apigen-mt": "APIGen-MT", "nemotron-rl-fc": "Nemotron-RL-Agentic-FC", "agent-flan": "Agent-FLAN",
    "agentinstruct": "AgentInstruct", "search-r1": "Search-R1 训练集", "asearcher": "ASearcher", "webshaper": "WebShaper",
    "swe-rl-pool": "SWE RL 任务池", "webchain": "WebChain", "sharerobot-plan": "ShareRobot (RoboBrain)",
}

# entry -> where to read one record. viewer: (hf, config, split[, offset]); file: (repo/path, bytes[, pick])
SRC = {
    "aguvis-s1": dict(file=("xlangai/aguvis-stage1/guienv.json", 40_000)),
    "aguvis-s2": dict(file=("xlangai/aguvis-stage2/amex-l3.json", 40_000)),
    "sharerobot-aff": dict(file=("BAAI/ShareRobot/affordance/affordance.json", 6_000)),
    "sharerobot-state": dict(file=("BAAI/ShareRobot/planning/train/success_(negative)_task.json", 20_000)),
    "sharerobot-afford": dict(file=("BAAI/ShareRobot/planning/train/discriminative_affordance_(negative)_task.json", 20_000)),
    "sharerobot-future": dict(file=("BAAI/ShareRobot/planning/train/future_prediction_task.json", 20_000)),
    "sharerobot-plan": dict(file=("BAAI/ShareRobot/planning/train/planning_with_context_task.json", 20_000)),
    "cosmos-understanding": dict(viewer=("nvidia/Cosmos-Reason1-SFT-Dataset", "bridgev2", "understanding")),
    "cosmos-reasoning": dict(viewer=("nvidia/Cosmos-Reason1-SFT-Dataset", "agibot", "reasoning")),
    "nemotron-agentic-v4": dict(file=("nvidia/Nemotron-SFT-Agentic-v4/data/knowledge_agent.jsonl", 400_000)),
    "swe-hero": dict(viewer=("nvidia/SWE-Hero-openhands-trajectories", None, None)),
    "swe-rebench-oh": dict(viewer=("nebius/SWE-rebench-openhands-trajectories", None, None)),
    "spar-7m": dict(viewer=("jasonzhango/SPAR-7M", None, None, 2000)),
    "eo-grounding": dict(viewer=("IPEC-COMMUNITY/EO-Data1.5M", "qa-trajectory_qa", "train")),
    "eo-state": dict(viewer=("IPEC-COMMUNITY/EO-Data1.5M", "qa-failure_detection", "train")),
    "eo-afford": dict(viewer=("IPEC-COMMUNITY/EO-Data1.5M", "qa-affordance_qa", "train")),
    "eo-physics": dict(viewer=("IPEC-COMMUNITY/EO-Data1.5M", "qa-physical_common_sense", "train")),
    "eo-plan": dict(viewer=("IPEC-COMMUNITY/EO-Data1.5M", "qa-task_planning", "train")),
    "eo-interleave": dict(viewer=("IPEC-COMMUNITY/EO-Data1.5M", "interleave-temporal", "train")),
    "grit": dict(viewer=("zzliang/GRIT", None, None)),
    "molmo2-videopoint": dict(viewer=("allenai/Molmo2-VideoPoint", "spatial reference", "train")),
    "groundcua": dict(file=("ServiceNow/GroundCUA/data/7-Zip/001f0079a489909eb94e47c2374b7bf36ab1842e314592ce30a34d18a54eb1df.json", 20_000)),
    "robo2vlm": dict(viewer=("keplerccc/Robo2VLM-1", "default", "train", 7)),
    "egoit": dict(viewer=("lmms-lab/EgoIT-99K", None, None)),
    "molmoact-mid": dict(viewer=("allenai/MolmoAct-Midtraining-Mixture", "molmoact_tabletop_primary", "train")),
    "molmoact-pre": dict(viewer=("allenai/MolmoAct-Pretraining-Mixture", "bridge_dataset", "train")),
    "ecot-bridge": dict(ecot=True),
    "action100m": dict(viewer=("facebook/action100m-preview", None, None)),
    "webworld": dict(file=("Qwen/WebWorldData/WebWorld_Training.jsonl", 300_000)),
    "cwm-swe": dict(viewer=("marin-community/swe-rebench-v2-CodeWorldModeling", "default", "traces")),
    "codeio": dict(file=("hkust-nlp/CodeIO-PyEdu-Reasoning/pyedur_full.jsonl", 60_000)),
    "py-state": dict(file=("Fraser/python-state-changes/data.jsonl", 4_000)),
    "bytesized32": dict(viewer=("thuml/bytesized32-world-model-cot", None, None)),
    "clevrer": dict(viewer=("zechen-nlp/clevrer", "counterfactual", "train")),
    "wma": dict(viewer=("LangAGI-Lab/world_model_for_wa_desc_with_tao_dataset", None, None)),
    "mobile-gui-wm": dict(file=("xwk123/Mobile-GUI-Worldmodel-SFT/trajectory_annotations/aitw_filtered-1.json", 40_000)),
    "scalecua-nav": dict(file=("OpenGVLab/ScaleCUA-Data/annotations/data_20250328_android_navigation_20250628.jsonl", 20_000)),
    "scalecua-ground": dict(file=("OpenGVLab/ScaleCUA-Data/annotations/data_20250310_ubuntu_action_grounding_20250407_202507011.jsonl", 20_000)),
    "agenttrek": dict(file=("xlangai/AgentTrek/agenttrek_1207.json", 120_000)),
    "coderforge": dict(viewer=("togethercomputer/CoderForge-Preview", "trajectories", "SWE_Smith")),
    "nemotron-terminal": dict(viewer=("nvidia/Nemotron-Terminal-Corpus", None, None)),
    "toolmind": dict(file=("Nanbeige/ToolMind/graph_syn_datasets/graphsyn.jsonl", 40_000)),
    "nemotron-agentic-v1": dict(file=("nvidia/Nemotron-Agentic-v1/data/interactive_agent.jsonl", 200_000)),
    "finevision": dict(viewer=("HuggingFaceM4/FineVision", "CoSyn_400k_chart", "train")),
    "nemotron-vlm-v2": dict(viewer=("nvidia/Nemotron-VLM-Dataset-v2", None, None)),
    "llava-ov15": dict(viewer=("mvp-lab/LLaVA-OneVision-1.5-Instruct-Data", None, None)),
    "cauldron": dict(viewer=("HuggingFaceM4/the_cauldron", "ai2d", "train")),
    "llava-video": dict(file=("lmms-lab/LLaVA-Video-178K/0_30_s_youtube_v0_1/0_30_s_youtube_mc_v0_1_qa_processed.json", 20_000)),
    "pixmo-docs": dict(viewer=("allenai/pixmo-docs", "charts", "train")),
}


def from_viewer(hid, config, split, offset, ctx):
    splits = get_json(f"{DS}/splits?dataset={urllib.parse.quote(hid)}")["splits"]
    s = next((x for x in splits if (config is None or x["config"] == config) and (split is None or x["split"] == split)), splits[0])
    q = urllib.parse.urlencode(dict(dataset=hid, config=s["config"], split=s["split"], offset=offset or 0, length=1))
    row = get_json(f"{DS}/rows?{q}", timeout=180)["rows"][0]["row"]
    return dict(kind="viewer", source=f"{hid} · {s['config']}/{s['split']} · row {offset or 0}",
                url=f"https://huggingface.co/datasets/{hid}/viewer/{urllib.parse.quote(s['config'])}/{s['split']}?row={offset or 0}",
                record=shrink(row, ctx), images=ctx["images"])


def from_file(path, nbytes, ctx, pick=None):
    recs = head_records(path, nbytes)
    if isinstance(recs, list) and len(recs) == 1 and isinstance(recs[0], list):
        recs = recs[0]
    idx = next((i for i, r in enumerate(recs) if pick(r)), 0) if pick else 0
    repo = "/".join(path.split("/")[:2])
    return dict(kind="file", source=f"{path} · 第 {idx + 1} 条", url=f"https://huggingface.co/datasets/{repo}",
                record=shrink(recs[idx], ctx), images=ctx["images"])


def ecot(ctx):
    """ECoT is one big dict keyed by episode path; show one step of one episode."""
    from tokens import http_range, scan
    head = http_range("Embodied-CoT/embodied_features_bridge", "embodied_features_bridge.json", 0, 200_000).decode("utf-8", "ignore")
    obj, _ = scan(head, start=r'"\d+": (\{"features")')[0]
    f = obj["features"]
    step = 10
    rec = {"episode_step": step, "language_instruction": obj.get("metadata", {}).get("language_instruction"),
           "move_primitive": f["move_primitive"][step], "gripper_position": f["gripper_position"][step],
           "bboxes(step)": f.get("bboxes", [None] * (step + 1))[step], "reasoning(step)": obj.get("reasoning", {}).get(str(step))}
    return dict(kind="file", source="embodied_features_bridge.json · 第 1 个 episode 的第 10 步（已从 episode 结构中摘出）",
                url="https://huggingface.co/datasets/Embodied-CoT/embodied_features_bridge", record=shrink(rec, ctx), images=[])


def sample_entry(key, e):
    ctx = dict(slug=slugify(key) + "-" + hashlib.md5(key.encode()).hexdigest()[:6], images=[])
    if e["name"] in ILLUSTRATIVE:
        ill = ILLUSTRATIVE[e["name"]]
        return dict(kind="illustrative", source=ill["why"], url=None, record=ill["record"], images=[])
    src = SRC.get(key)
    if src is None and key in REUSE and REUSE[key] in OLD:
        return OLD[REUSE[key]]
    if src is None:
        raise KeyError("no sample source")
    for old in IMG_DIR.glob(ctx["slug"] + "_*.jpg"):
        old.unlink()
    if "viewer" in src:
        v = src["viewer"]
        return from_viewer(v[0], v[1], v[2], v[3] if len(v) > 3 else 0, ctx)
    if "file" in src:
        return from_file(src["file"][0], src["file"][1], ctx, src.get("pick"))
    if src.get("ecot"):
        return ecot(ctx)
    raise KeyError("bad source")


def main(only):
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key, e in catalog.ENTRIES.items():
        if only and key not in only:
            continue
        if not only and key in out:
            continue
        try:
            out[key] = sample_entry(key, e)
            print(f"OK   {out[key]['kind']:<12} {key}  imgs={len(out[key]['images'])}", flush=True)
        except Exception as ex:
            print(f"FAIL {key}: {type(ex).__name__}: {str(ex)[:160]}", flush=True)
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    missing = [k for k in catalog.ENTRIES if k not in out]
    print(f"{len(out)} samples; missing: {missing}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))

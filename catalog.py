"""Direction taxonomy (数据分类.png) and the dataset catalog used by tokens.py / mix.py / build.py.

Each ENTRY is one cleanable unit: one producer, one format. Keys are stable ids.
  dir      primary direction (A-G); also: secondary directions it contributes to
  files    {hf_repo: [regex over repo paths]} — the text annotation files to count
           (images/videos live in archives and are counted via placeholders / path fields)
  single   True = large single-source dataset (>=500k samples or >=1B tokens, one coherent format)
  role     short note on what the entry gives this direction
"""

DIRECTIONS = [
    dict(id="A", en="Grounding", zh="空间与数值对齐", q="目标在哪里？在哪个坐标系？差多少？",
         out="点、框、位姿、变换、带单位的数值与容差"),
    dict(id="B", en="Understanding", zh="历史与任务状态理解", q="过去发生了什么？现在处于什么状态？",
         out="事件、物体状态、进度、证据、不确定性"),
    dict(id="C", en="Identification", zh="动作语义与接口辨识", q="当前 action / API 到底如何作用于系统？",
         out="接口解释、控制映射、假设与辨识结果"),
    dict(id="D", en="WorldModeling", zh="动作条件的环境建模", q="执行候选动作会导致什么变化？",
         out="下一状态、事件、接触、风险、分支结果"),
    dict(id="E", en="Decision", zh="目标条件的决策与恢复", q="为实现目标，下一步该做什么？失败如何修正？",
         out="动作、子目标、计划、停止 / 追问 / 探测决策"),
    dict(id="F", en="Agentic", zh="程序化闭环执行", q="如何通过代码和工具完成并检查任务？",
         out="调用、程序、反馈处理、修复、再次执行"),
    dict(id="G", en="General", zh="通用视觉语言能力", q="能否准确看懂、读懂并回答一般多模态问题？",
         out="描述、问答、OCR、图表理解、一般视觉推理"),
]

ALL = r".*\.(json|jsonl|parquet)"

# Visual tokens. Qwen3.x ViT: 16 px patch, 2x2 merge -> 1 token per 32x32 px.
# Images: native resolution clamped to [65,536, 1,048,576] px  -> 64..1024 tokens.
# Videos: 16 sampled frames, temporal patch 2 -> 8 groups; each frame capped at 448x448 px (196 tokens).
def img_tok(w, h):
    return round(min(max(w * h, 65536), 1048576) / 1024)


def vid_tok(w, h, frames=16):
    return round(frames / 2 * min(w * h, 200704) / 1024)


SCREEN = img_tok(1080, 2400)  # phone / desktop screenshots hit the 1024 cap

ENTRIES = {}


def E(key, **kw):
    kw.setdefault("also", [])
    kw.setdefault("single", False)
    kw.setdefault("files", {})
    kw.setdefault("links", [])
    kw.setdefault("note", "")
    kw.setdefault("tier", "core")
    kw.setdefault("vtok", dict(image=img_tok(896, 896), video=vid_tok(640, 480)))
    kw.setdefault("count_images", kw.get("modality") != "text")  # stray "*.png" strings in tool logs are not images
    ENTRIES[key] = kw


# ---------------------------------------------------------------- A Grounding
E("pixmo-points", dir="A", name="PixMo-Points", hf=["allenai/pixmo-points"], single=True,
  files={"allenai/pixmo-points": [r"data/.*\.parquet"]}, modality="image", vtok=dict(image=img_tok(962, 683)),
  role="230 万条指点 / 计数标注，单一格式（图像 URL + 点坐标），最大的公开 pointing 数据")
E("os-atlas", dir="A", name="OS-Atlas grounding", hf=["OS-Copilot/OS-Atlas-data"], single=True,
  files={"OS-Copilot/OS-Atlas-data": [r"(web|mobile|desktop)_domain/.*\.json"]}, modality="image", vtok=dict(image=SCREEN),
  note="每条 = 一张截图 + 多个元素；与 Aguvis stage1 的 seeclick / widget 部分有重叠，二者只取其一",
  role="1300 万+ GUI 元素 (指令 → bbox)，跨 web / mobile / desktop，同一 schema")
E("robopoint", dir="A", name="RoboPoint", hf=["wentao-yuan/robopoint-data"], single=True,
  files={"wentao-yuan/robopoint-data": [r"robopoint_1432k\.json"]}, modality="image", vtok=dict(image=img_tok(640, 480)),
  role="143 万条空间 affordance 点预测（物体参照 / 空闲区域）")
E("refspatial", dir="A", name="RefSpatial (RoboRefer)", hf=["JingkunAn/RefSpatial"], single=True,
  files={"JingkunAn/RefSpatial": [r"(2D|3D)/.*\.json", r"Simulator/metadata\.json"]}, modality="image", vtok=dict(image=img_tok(900, 640)),
  note="深度图不计 token（作为辅助输入时另算）",
  role="多步空间指代推理 + 深度，2D/3D/仿真三部分同一格式")
E("uground", dir="A", name="UGround", hf=["osunlp/UGround-V1-Data"], single=True,
  files={"osunlp/UGround-V1-Data": [r".*\.parquet"]}, modality="image", note="gated", vtok=dict(image=SCREEN),
  role="1000 万 web 元素指代表达 → 坐标")
E("aguvis-s1", dir="A", name="Aguvis stage1 (grounding)", hf=["xlangai/aguvis-stage1"],
  files={"xlangai/aguvis-stage1": [r"(guienv|ricoig16k|ui_refexp|webui350k)\.json"]}, modality="image", vtok=dict(image=SCREEN),
  note="只计 OS-Atlas 未覆盖的子集（guienv / ricoig16k / ui_refexp / webui350k）；seeclick / ricosca / widget 已在 OS-Atlas 中",
  role="多源 GUI grounding 统一成 pyautogui 动作格式")
E("vsi-590k", dir="A", also=["B"], name="VSI-590K (Cambrian-S)", hf=["nyu-visionx/VSI-590K"], single=True,
  files={"nyu-visionx/VSI-590K": [r"vsi_590k\.jsonl"]}, modality="video", vtok=dict(image=0, video=vid_tok(640, 480, 32)),
  count_images=False, note="每条是一段 ScanNet / ARKit 等室内扫描视频（<image> 占位实为视频帧，按 32 帧计）",
  role="59 万视频空间 QA：距离、尺寸、相对方位、路线（带度量答案）")
E("sharerobot-aff", dir="A", name="ShareRobot affordance / trajectory", hf=["BAAI/ShareRobot"],
  files={"BAAI/ShareRobot": [r"affordance/affordance\.json", r"trajectory/trajectory\.json"]}, modality="image", vtok=dict(image=img_tok(424, 256)),
  role="OXE 帧上的 affordance 框与末端 2D 轨迹")

# ------------------------------------------------------------ B Understanding
CR = "nvidia/Cosmos-Reason1-SFT-Dataset"
E("cosmos-understanding", dir="B", name="Cosmos-Reason1：视频理解", hf=[CR], single=True, group="cosmos",
  files={CR: [r".*/.*_understanding\.json"]}, modality="video", vtok=dict(video=vid_tok(640, 512)),
  role="RoboVQA / BridgeV2 / AgiBot / HoloAssist 视频中的物体与动作稠密描述（52 万条）")
E("cosmos-reasoning", dir="E", also=["B"], name="Cosmos-Reason1：下一步动作推理", hf=[CR], single=True, group="cosmos",
  files={CR: [r"(agibot|bridgev2|holoassist)/.*_reasoning\.json", r"robovqa/robovqa_reasoning_0\.json"]}, modality="video",
  vtok=dict(video=vid_tok(640, 512)), note="robovqa_reasoning_1-4 是同一批题的 4 次重采样 CoT，只计 reasoning_0 一份",
  role="长 CoT：视频中智能体在做什么、下一步该做什么（<think> + <answer>）")
SR = "planning/train/"
E("sharerobot-state", dir="B", name="ShareRobot：历史与成功判断", hf=["BAAI/ShareRobot"], single=True, group="sharerobot",
  files={"BAAI/ShareRobot": [SR + r"(past_description|success_\((positive|negative)\))_task\.json"]}, modality="image",
  vtok=dict(image=img_tok(424, 256)), role="OXE 多帧：最近完成了什么动作、子任务是否完成（正 / 负样本）")
E("sharerobot-afford", dir="C", name="ShareRobot：动作可行性辨识", hf=["BAAI/ShareRobot"], single=True, group="sharerobot",
  files={"BAAI/ShareRobot": [SR + r"(discriminative_affordance_\((positive|negative)\)|generative_affordance)_task\.json"]}, modality="image",
  vtok=dict(image=img_tok(424, 256)), role="当前状态下某个动作能否执行 / 现在能执行哪个动作：动作语义与前置条件辨识")
E("sharerobot-future", dir="D", name="ShareRobot：下一事件预测", hf=["BAAI/ShareRobot"], single=True, group="sharerobot",
  files={"BAAI/ShareRobot": [SR + r"future_prediction_task\.json"]}, modality="image",
  vtok=dict(image=img_tok(424, 256)), role="执行完某个动作后最可能发生的下一事件：动作条件的状态预测")
E("sharerobot-plan", dir="E", name="ShareRobot：规划与剩余步骤", hf=["BAAI/ShareRobot"], single=True, group="sharerobot",
  files={"BAAI/ShareRobot": [SR + r"planning(_with_context|_remaining_steps)?_task\.json"]}, modality="image",
  vtok=dict(image=img_tok(424, 256)), role="给定目标与已完成步骤，下一步 / 剩余步骤规划")
E("robovqa", dir="B", also=["E"], name="RoboVQA", hf=["Tianli/robovqa"],
  files={"Tianli/robovqa": [r"json/train/.*\.json"]}, modality="video", vtok=dict(video=vid_tok(640, 512)),
  role="长程视频：成功检测、可行性、规划、下一步预测（HF 社区转换版）")
E("robofac", dir="B", also=["E"], name="RoboFAC", hf=["MINT-SJTU/RoboFAC-dataset"],
  files={"MINT-SJTU/RoboFAC-dataset": [r"training_qa\.json"]}, modality="video", vtok=dict(video=vid_tok(512, 512)),
  role="失败视频 8 类 QA：检测、定位、类型、原因、纠正")
E("aha", dir="B", name="AHA", hf=["ACIDE/AHA_Dataset_2e", "ACIDE/AHA_Dataset_2e_1p", "ACIDE/AHA_Dataset_2e_2p"],
  files={r: [r"data/.*\.parquet"] for r in ["ACIDE/AHA_Dataset_2e", "ACIDE/AHA_Dataset_2e_1p", "ACIDE/AHA_Dataset_2e_2p"]},
  modality="image", vtok=dict(image=img_tok(256, 256)), role="RLBench 程序化失败注入：多视角帧 + 失败原因")

# ----------------------------------------------------------- C Identification
E("toucan-c", dir="C", also=["F"], name="Toucan-1.5M（工具 schema / 调用语义）", hf=["Agent-Ark/Toucan-1.5M"],
  files={}, modality="text", tier="ref", note="token 计入 F；这里只作为 API 语义来源列出",
  role="~500 个真实 MCP server 的工具定义与调用，可抽取“工具做什么 / 参数含义”")
E("when2call", dir="C", also=["E"], name="When2Call", hf=["nvidia/When2Call"],
  files={"nvidia/When2Call": [r"train/.*\.jsonl"]}, modality="text",
  role="判断工具能否作用于当前请求：可调 / 缺参追问 / 无合适工具")
E("hermes-fc", dir="C", also=["F"], name="Hermes Function Calling", hf=["NousResearch/hermes-function-calling-v1"],
  files={"NousResearch/hermes-function-calling-v1": [r".*\.json"]}, modality="text",
  role="函数签名 → 调用 / JSON 模式，接口映射的基础格式")
E("xlam", dir="C", also=["F"], name="xLAM-60k (APIGen)", hf=["Salesforce/xlam-function-calling-60k"],
  files={"Salesforce/xlam-function-calling-60k": [r".*\.json"]}, modality="text", note="gated",
  role="三级校验的 query → API 调用映射")
E("toolace", dir="C", also=["F"], name="ToolACE", hf=["Team-ACE/ToolACE"],
  files={"Team-ACE/ToolACE": [r"data\.json"]}, modality="text",
  role="26k 合成 API 的多样接口与参数语义")

# ------------------------------------------------------------ D WorldModeling
# (filled mostly from new candidates; see NEW_ENTRIES below)

# ----------------------------------------------------------------- E Decision
E("aguvis-s2", dir="E", also=["A"], name="Aguvis stage2 (GUI 规划)", hf=["xlangai/aguvis-stage2"], single=True,
  files={"xlangai/aguvis-stage2": [r".*-l3\.json", r"(android_control|coat|guide|guiact-web-single)\.json"]}, modality="image", vtok=dict(image=SCREEN),
  note="l1 / l2 / l3 是同一批步骤的三种推理详略，只计信息最全的 l3；已包含 AndroidControl、GUI-Odyssey、AITW、AMEX、Mind2Web",
  role="AITW / AMEX / AndroidControl / GUI-Odyssey / Mind2Web 等统一成“观察 → 思考 → 动作”格式")
E("agentnet", dir="E", also=["F"], name="AgentNet (OpenCUA)", hf=["xlangai/AgentNet"],
  files={"xlangai/AgentNet": [r"agentnet_.*\.jsonl"]}, modality="image", vtok=dict(image=SCREEN),
  role="2.3 万桌面长程轨迹，带反思式 CoT")
E("androidcontrol", dir="E", name="AndroidControl", hf=["ckg/AndroidControlParsedWithImages-20k"],
  files={"ckg/AndroidControlParsedWithImages-20k": [r"data/.*\.parquet"]}, modality="image", vtok=dict(image=SCREEN), tier="dup",
  note="已包含在 Aguvis stage2 中，不重复计入配比",
  role="高 / 低层指令双标注的手机操作步")
E("gui-odyssey", dir="E", name="GUI-Odyssey", hf=["OpenGVLab/GUI-Odyssey"],
  files={"OpenGVLab/GUI-Odyssey": [r"all_anno\.json"]}, modality="image", vtok=dict(image=SCREEN), tier="dup",
  note="已包含在 Aguvis stage2 中，不重复计入配比",
  role="跨 App 手机长程任务")
E("embodied-reasoner", dir="E", also=["B"], name="Embodied-Reasoner", hf=["zwq2018/embodied_reasoner"],
  files={"zwq2018/embodied_reasoner": [r"train_multiturn_9390\.json"]}, modality="image", vtok=dict(image=img_tok(500, 500)),
  role="AI2-THOR 交互式搜索，带反思与自我纠错")
E("ebench", dir="E", name="EmbodiedBench 轨迹", hf=["EmbodiedBench/EB-Man_trajectory_dataset", "EmbodiedBench/EB-Alfred_trajectory_dataset", "EmbodiedBench/EB-Nav_trajectory_dataset"],
  files={r: [r".*multi_step\.json"] for r in ["EmbodiedBench/EB-Man_trajectory_dataset", "EmbodiedBench/EB-Alfred_trajectory_dataset", "EmbodiedBench/EB-Nav_trajectory_dataset"]},
  modality="image", vtok=dict(image=img_tok(500, 500)), role="多模型在 EB-ALFRED / Nav / Manipulation 的成功与失败轨迹")
E("era", dir="E", name="ERA 先验数据", hf=["EmbodiedReasoningAgent/EB-ALFRED_trajectory_augmented_prior_dataset", "EmbodiedReasoningAgent/EB-ALFRED_environment_anchored_prior_dataset",
                                        "EmbodiedReasoningAgent/EB-ALFRED_external_knowledge_prior_dataset", "EmbodiedReasoningAgent/EB-Man_trajectory_augmented_prior_dataset",
                                        "EmbodiedReasoningAgent/EB-Man_environment_anchored_prior_dataset", "EmbodiedReasoningAgent/EB-Man_external_knowledge_prior_dataset"],
  files={r: [r".*\.json"] for r in ["EmbodiedReasoningAgent/EB-ALFRED_trajectory_augmented_prior_dataset", "EmbodiedReasoningAgent/EB-ALFRED_environment_anchored_prior_dataset",
                                    "EmbodiedReasoningAgent/EB-ALFRED_external_knowledge_prior_dataset", "EmbodiedReasoningAgent/EB-Man_trajectory_augmented_prior_dataset",
                                    "EmbodiedReasoningAgent/EB-Man_environment_anchored_prior_dataset", "EmbodiedReasoningAgent/EB-Man_external_knowledge_prior_dataset"]},
  modality="image", vtok=dict(image=img_tok(500, 500)), role="轨迹增强 / 环境锚定 / 外部知识三类先验")
E("agenttraj-l", dir="E", also=["F"], name="AgentGym AgentTraj-L", hf=["AgentGym/AgentTraj-L"],
  files={"AgentGym/AgentTraj-L": [r".*_train\.json"]}, modality="text",
  role="14 个文本环境（ALFWorld / WebShop / SciWorld…）的专家轨迹")
E("egoplan", dir="E", name="EgoPlan-Bench2", hf=["qiulu66/EgoPlan-Bench2"], tier="eval",
  files={}, modality="video", note="评测集，不进训练配比",
  role="第一视角下一步规划（评测）")

# ------------------------------------------------------------------ F Agentic
E("toucan", dir="F", also=["C"], name="Toucan-1.5M", hf=["Agent-Ark/Toucan-1.5M"], single=True,
  files={"Agent-Ark/Toucan-1.5M": [r"(Kimi-K2|Qwen3)/.*\.parquet"]}, modality="text",
  note="Kimi-K2 与 OSS 两个 config 约一半是同一批问题；只计 Kimi-K2 + Qwen3（问题几乎不重叠），SFT config 是三者的精选子集",
  role="165 万条真实 MCP 工具使用轨迹，同一 messages 格式")
E("nemotron-agentic-v2", dir="F", name="Nemotron-SFT-Agentic-v2", hf=["nvidia/Nemotron-SFT-Agentic-v2"], single=True,
  files={"nvidia/Nemotron-SFT-Agentic-v2": [r"data/.*\.jsonl"]}, modality="text",
  role="NVIDIA 工业级多轮 / 并行工具调用与交互式 agent 轨迹")
E("nemotron-agentic-v4", dir="F", name="Nemotron-SFT-Agentic-v4", hf=["nvidia/Nemotron-SFT-Agentic-v4"], single=True,
  files={"nvidia/Nemotron-SFT-Agentic-v4": [r"data/.*\.jsonl"]}, modality="text",
  role="v4：tool_calling + knowledge_agent")
E("swe-zero", dir="F", name="NVIDIA SWE-Zero", hf=["nvidia/SWE-Zero-openhands-trajectories"], single=True,
  files={"nvidia/SWE-Zero-openhands-trajectories": [r"data/.*\.parquet"]}, modality="text",
  role="32 万条 OpenHands SWE 轨迹（免执行），同一格式")
E("swe-hero", dir="F", name="NVIDIA SWE-Hero", hf=["nvidia/SWE-Hero-openhands-trajectories"],
  files={"nvidia/SWE-Hero-openhands-trajectories": [r"data/.*\.parquet"]}, modality="text",
  role="基于真实执行反馈的 SWE 轨迹")
E("swe-smith-traj", dir="F", name="SWE-smith 轨迹", hf=["SWE-bench/SWE-smith-trajectories"],
  files={"SWE-bench/SWE-smith-trajectories": [r"data/tool-.*\.parquet"]}, modality="text",
  note="只计 tool 格式一份（xml / ticks / train 是同一批轨迹的不同渲染）", role="合成 bug + 修复轨迹")
E("swe-rebench-oh", dir="F", name="SWE-rebench OpenHands 轨迹", hf=["nebius/SWE-rebench-openhands-trajectories"],
  files={"nebius/SWE-rebench-openhands-trajectories": [r"trajectories\.parquet"]}, modality="text",
  role="持续更新的防污染 SWE 轨迹")
E("nebius-swe-agent", dir="F", name="Nebius SWE-agent 轨迹", hf=["nebius/SWE-agent-trajectories"],
  files={"nebius/SWE-agent-trajectories": [r"data/.*\.parquet"]}, modality="text",
  role="8 万条，成功 + 失败均有")
E("r2e-gym", dir="F", name="R2E-Gym SFT / Verifier", hf=["R2E-Gym/R2EGym-SFT-Trajectories", "R2E-Gym/R2EGym-Verifier-Trajectories"],
  files={"R2E-Gym/R2EGym-SFT-Trajectories": [ALL], "R2E-Gym/R2EGym-Verifier-Trajectories": [ALL]}, modality="text",
  role="SFT 轨迹 + 验证器轨迹")
E("adp", dir="F", name="ADP (Agent Data Protocol)", hf=["neulab/agent-data-collection"], tier="ref",
  files={}, modality="text", note="多源合集，含 Toucan / CoderForge / SWE-smith 等的重复；不计入配比以免重复计数",
  role="23 个子集统一 std 格式，适合当格式转换中间层")
E("openresearcher", dir="F", name="OpenResearcher", hf=["OpenResearcher/OpenResearcher-Dataset"],
  files={"OpenResearcher/OpenResearcher-Dataset": [r"seed_42/.*\.parquet"]}, modality="text",
  note="16 个 seed 是同一题集的重复采样，只计 seed_42 一份", role="长程 deep research 轨迹")
E("miroverse", dir="F", name="MiroVerse", hf=["miromind-ai/MiroVerse-v0.1"],
  files={"miromind-ai/MiroVerse-v0.1": [r"jsonl_sft/.*\.jsonl"]}, modality="text", note="gated",
  role="多源 deep research SFT 轨迹")
E("toolbench", dir="F", name="ToolBench (DFS)", hf=["Yhyu13/ToolBench_toolllama_G123_dfs"],
  files={"Yhyu13/ToolBench_toolllama_G123_dfs": [r"toolllama_G123_dfs_train\.json"]}, modality="text",
  role="DFSDT 搜索树，含失败回溯")
E("apigen-mt", dir="F", name="APIGen-MT-5k", hf=["Salesforce/APIGen-MT-5k"],
  files={"Salesforce/APIGen-MT-5k": [ALL]}, modality="text", role="τ-bench 风格多轮 agent-用户交互")
E("nemotron-rl-fc", dir="F", name="Nemotron-RL-Agentic-FC", hf=["nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1"], tier="rl",
  files={"nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1": [ALL]}, modality="text", role="工具调用 RL 任务")
E("agent-flan", dir="F", name="Agent-FLAN", hf=["internlm/Agent-FLAN"],
  files={"internlm/Agent-FLAN": [r"data/.*\.jsonl"]}, modality="text", role="格式 / 推理解耦 + 负样本")
E("agentinstruct", dir="F", name="AgentInstruct", hf=["zai-org/AgentInstruct"],
  files={"zai-org/AgentInstruct": [ALL]}, modality="text", role="早期 6 环境 agent 轨迹")
E("search-r1", dir="F", name="Search-R1 NQ+HotpotQA", hf=["PeterJinGo/nq_hotpotqa_train"], tier="rl",
  files={"PeterJinGo/nq_hotpotqa_train": [r"train\.parquet"]}, modality="text", role="搜索 RL 起点（只有问答，无轨迹）")
E("asearcher", dir="F", name="ASearcher", hf=["inclusionAI/ASearcher-train-data"], tier="rl",
  files={"inclusionAI/ASearcher-train-data": [ALL]}, modality="text", role="高难 QA，搜索 RL 用")
E("webshaper", dir="F", name="WebShaper", hf=["Alibaba-NLP/WebShaper"], tier="rl",
  files={"Alibaba-NLP/WebShaper": [ALL]}, modality="text", role="形式化合成 IS 任务（500 题）")
E("swe-rl-pool", dir="F", name="SWE RL 任务池", hf=["SWE-bench/SWE-smith", "R2E-Gym/R2E-Gym-Subset", "nebius/SWE-rebench", "SWE-Gym/SWE-Gym", "nebius/SWE-rebench-V2"],
  tier="rl", files={}, modality="text", note="RL 环境实例，不是 SFT token", role="可执行任务实例")
E("webchain", dir="E", name="WebChain", hf=["webagentlab/webchain"], tier="ref",
  files={}, modality="image", note="386 GB 原始 parquet，token 未统计；作补充",
  role="3.2 万人工真实网站轨迹")


# =====================================================================================
# New candidates (2026-10-24): large single-source sets that fill A–G, esp. C / D / G
# =====================================================================================
EO = "IPEC-COMMUNITY/EO-Data1.5M"
EO_V = dict(image=img_tok(364, 252))  # EO-Data stores AgiBot / OXE frames at 364x252 (viewer-measured)
FV = "HuggingFaceM4/FineVision"
# FineVision configs that duplicate GUI / embodied data counted in A / E
FV_SKIP = r"(aguvis-stage-1|groundui|screenqa|screen2words|alfworldgpt|websight|drivelm)"

# ---- A Grounding
E("spar-7m", dir="A", also=["D"], name="SPAR-7M", hf=["jasonzhango/SPAR-7M"], single=True, priority=1,
  files={"jasonzhango/SPAR-7M": [r"spar-dataset-train\.parquet"]}, modality="image", vtok=dict(image=img_tok(640, 480)),
  links=[("arXiv 2503.22976", "https://arxiv.org/abs/2503.22976")], new=True,
  note="只有 QA；RGB-D 图像在 jasonzhango/SPAR-7M-RGBD", role="1600 万条 3D 空间 QA（33 类任务，多视角 / 度量深度 / 距离 / 导航），一条管线产出")
E("eo-grounding", dir="A", name="EO-Data1.5M：点 / 框 / 轨迹 / 多视角对应", hf=[EO], single=True, group="eo",
  files={EO: [r"qa-(points_qa|object_referring_qa|trajectory_qa|multiview_qa)/.*\.parquet"]}, modality="image", vtok=EO_V, new=True,
  links=[("arXiv 2508.21112", "https://arxiv.org/abs/2508.21112")], role="AgiBot / OXE 帧上的接触点、物体框、末端 2D 轨迹、跨视角点对应")
E("grit", dir="A", name="GRIT-20M (Kosmos-2)", hf=["zzliang/GRIT"], single=True, priority=8,
  files={"zzliang/GRIT": [r"grit-20m/.*\.parquet"]}, modality="image", vtok=dict(image=img_tok(736, 1050)), new=True,
  note="只给图像 URL，需要回源下载；网页图像，与具身场景差距大，排在具身 grounding 之后",
  role="2000 万对 名词短语 / 指代表达 → 框")
E("molmo2-videopoint", dir="A", name="Molmo2-VideoPoint", hf=["allenai/Molmo2-VideoPoint"], new=True,
  files={"allenai/Molmo2-VideoPoint": [r"(?!data/).*/train-.*\.parquet"]}, modality="video", vtok=dict(video=vid_tok(640, 360)),
  note="只给 YouTube ID；data/ 是其余子集的合并，不重复计", role="视频中的指点与计数（130 万条）")
E("groundcua", dir="A", name="GroundCUA", hf=["ServiceNow/GroundCUA"], new=True,
  files={}, modality="image", tier="ref", vtok=dict(image=SCREEN),
  note="5 万个逐截图 JSON 小文件，token 未统计；与 OS-Atlas 同类，作补充", role="87 个桌面平台人工标注的 GUI 元素框")

# ---- B Understanding
E("eo-state", dir="B", name="EO-Data1.5M：完成判断 / 过程校验 / 片段描述", hf=[EO], single=True, group="eo",
  files={EO: [r"qa-(failure_detection|subtask_qa|process_verification|episode_caption)/.*\.parquet"]}, modality="image", vtok=EO_V, new=True,
  role="子任务是否完成、最近完成了哪一步、当前画面描述（AgiBot / OXE 真机）")
E("robo2vlm", dir="B", also=["A", "D"], name="Robo2VLM-1", hf=["keplerccc/Robo2VLM-1"], single=True, new=True,
  files={"keplerccc/Robo2VLM-1": [r"data/train-.*\.parquet"]}, modality="image", vtok=dict(image=img_tok(640, 360)),
  links=[("arXiv 2505.15517", "https://arxiv.org/abs/2505.15517")],
  role="68 万道由 OXE / DROID 轨迹生成的选择题：任务阶段与成功、深度、多视角、运动方向")
E("egoit", dir="B", name="EgoIT-99K", hf=["lmms-lab/EgoIT-99K"], new=True,
  files={"lmms-lab/EgoIT-99K": [r"parquet/.*\.parquet"]}, modality="video", vtok=dict(video=vid_tok(640, 480)),
  role="第一视角视频指令数据（Ego4D / EPIC / HoloAssist 等 9 源统一格式）")

# ---- C Identification
E("molmoact-mid", dir="C", also=["A", "E"], name="MolmoAct Mid-training Mixture", hf=["allenai/MolmoAct-Midtraining-Mixture"], single=True, priority=1,
  files={"allenai/MolmoAct-Midtraining-Mixture": [r".*\.parquet"]}, modality="image", vtok=dict(image=img_tok(640, 480)), new=True,
  links=[("arXiv 2508.07917", "https://arxiv.org/abs/2508.07917")],
  note="每条两张图（主视角 + 腕部）；深度与动作以离散 token 写在文本里",
  role="590 万条动作推理链：任务 → 深度感知 token → 末端轨迹 → 动作 token，同一产出方同一格式")
E("molmoact-pre", dir="C", also=["A"], name="MolmoAct Pre-training Mixture", hf=["allenai/MolmoAct-Pretraining-Mixture"], single=True, priority=2,
  files={"allenai/MolmoAct-Pretraining-Mixture": [r"(bc_z|bridge_dataset|fractal20220817_data|auxiliary_trace)/.*\.parquet"]},
  modality="image", vtok=dict(image=img_tok(285, 228)), new=True,
  note="OXE 三个子集（bc_z / bridge / fractal）上的同格式动作推理 + 轨迹辅助任务；auxiliary_depth 与 lvis 不计",
  role="2400 万条：OXE 帧 → 深度 token → 轨迹 → 动作（动作语义与控制映射）")
E("ecot-bridge", dir="C", also=["E"], name="Embodied-CoT (Bridge)", hf=["Embodied-CoT/embodied_features_bridge"], new=True,
  files={"Embodied-CoT/embodied_features_bridge": [r"embodied_features_bridge\.json"]}, modality="image",
  vtok=dict(image=0), count_images=False, links=[("arXiv 2407.08693", "https://arxiv.org/abs/2407.08693")],
  json_start=r'"\d+": (\{"features")',
  note="按 episode 存逐步标注；图像在 Bridge V2 RLDS 中，未计视觉 token",
  role="Bridge V2 每一步的运动原语（move left / close gripper）、夹爪位置、物体框、子任务及理由")
E("eo-afford", dir="C", name="EO-Data1.5M：动作可行性", hf=[EO], single=True, group="eo",
  files={EO: [r"qa-affordance_qa/.*\.parquet"]}, modality="image", vtok=EO_V, new=True,
  role="当前状态下某个动作能否立即执行")
E("action100m", dir="C", also=["B"], name="Action100M (preview)", hf=["facebook/action100m-preview"], new=True, priority=8,
  files={"facebook/action100m-preview": [r"data/.*\.parquet"]}, modality="video", vtok=dict(video=vid_tok(640, 360)),
  count_images=False, tier="ref", note="FAIR 非商用许可；只给 YouTube ID；全量的 10%。每个视频是一棵 PLM / Llama3 / GPT 三套描述的节点树（原始标注，非 SFT 样本），全文约 27B token，需先抽取成 动作片段 → 描述 再用",
  role="12 万个视频、约 1000 万个开放词表动作片段与层级描述")

# ---- D WorldModeling
E("webworld", dir="D", name="Qwen WebWorldData", hf=["Qwen/WebWorldData"], single=True, priority=1,
  files={"Qwen/WebWorldData": [r"WebWorld_Training\.jsonl"]}, modality="text", new=True,
  role="web 世界模型：给定页面 a11y 树与动作序列，预测每一步之后的完整页面状态（106 万条轨迹）")
E("cwm-swe", dir="D", name="SWE-rebench-V2 CodeWorldModeling", hf=["marin-community/swe-rebench-v2-CodeWorldModeling"], single=True, priority=2,
  files={"marin-community/swe-rebench-v2-CodeWorldModeling": [r"data/traces-.*\.parquet"]}, modality="text", new=True,
  note="CC-BY-4.0，另须遵守上游仓库许可", role="186 万条逐行执行 trace（修复前 / 后两次运行）：程序执行的世界模型")
E("codeio", dir="D", name="CodeI/O PyEdu-Reasoning", hf=["hkust-nlp/CodeIO-PyEdu-Reasoning"], single=True, priority=3,
  files={"hkust-nlp/CodeIO-PyEdu-Reasoning": [r"pyedur_full\.jsonl"]}, modality="text", new=True,
  links=[("arXiv 2502.07316", "https://arxiv.org/abs/2502.07316")], role="给定代码与输入预测输出（或反推输入）+ CoT")
E("eo-physics", dir="D", name="EO-Data1.5M：物理常识 / 关系推理", hf=[EO], single=True, group="eo",
  files={EO: [r"qa-(physical_common_sense|relation_reasoning)/.*\.parquet"]}, modality="image", vtok=EO_V, new=True,
  role="动作对场景的物理影响（重心、接触、角度）")
E("py-state", dir="D", name="python-state-changes", hf=["Fraser/python-state-changes"], single=True, priority=6,
  files={"Fraser/python-state-changes": [r"data\.jsonl"]}, modality="text", new=True,
  note="卡片未写许可", role="906 万条 {变量状态, 一行代码, 执行后状态}")
E("bytesized32", dir="D", name="ByteSized32 world-model CoT", hf=["thuml/bytesized32-world-model-cot"], new=True,
  files={"thuml/bytesized32-world-model-cot": [r".*\.parquet"]}, modality="text",
  role="文本游戏 JSON 状态 + 动作 → 下一状态（带 RL ground truth）")
E("clevrer", dir="D", also=["B"], name="CLEVRER (predictive / counterfactual)", hf=["zechen-nlp/clevrer"], new=True,
  files={"zechen-nlp/clevrer": [r"(predictive|counterfactual)/train-.*\.parquet"]}, modality="video", vtok=dict(video=vid_tok(480, 320)),
  count_images=False, note="卡片未写许可，应按上游 CLEVRER", role="碰撞事件预测与反事实（移除某物体后会发生什么）")
E("wma", dir="D", name="WMA WebArena 状态差异", hf=["LangAGI-Lab/world_model_for_wa_desc_with_tao_dataset"], new=True,
  files={"LangAGI-Lab/world_model_for_wa_desc_with_tao_dataset": [r"data/.*\.parquet"]}, modality="text",
  role="WebArena 中动作后的下一状态差异描述（质量高、规模小）")
E("mobile-gui-wm", dir="D", name="Mobile GUI World Model SFT", hf=["xwk123/Mobile-GUI-Worldmodel-SFT"], new=True,
  files={"xwk123/Mobile-GUI-Worldmodel-SFT": [r"trajectory_annotations/.*\.json"]}, modality="image", vtok=dict(image=SCREEN),
  role="手机 GUI 动作后的下一页面状态（AITW / AMEX / AndroidControl）")

# ---- E Decision
E("eo-plan", dir="E", name="EO-Data1.5M：任务规划", hf=[EO], single=True, group="eo",
  files={EO: [r"qa-task_planning/.*\.parquet"]}, modality="image", vtok=EO_V, new=True,
  role="给定目标与已完成子任务，预测下一子任务（AgiBot / OXE 真机）")
E("eo-interleave", dir="E", also=["B", "C"], name="EO-Data1.5M：交错推理-动作", hf=[EO], single=True, group="eo",
  files={EO: [r"interleave-(temporal|trajectory|free_chat|random_qa)/.*\.parquet"]}, modality="image", vtok=EO_V, new=True,
  note="交错样本里的 <action> 是连续动作块（VLA 训练用），这里只计文本与图像 token",
  role="同一 episode 内交错的 规划 → 执行 → 完成校验 多轮对话")
E("scalecua-nav", dir="E", name="ScaleCUA 导航轨迹", hf=["OpenGVLab/ScaleCUA-Data"], new=True,
  files={"OpenGVLab/ScaleCUA-Data": [r"annotations/.*navigation.*\.jsonl"]}, modality="image", vtok=dict(image=SCREEN),
  links=[("arXiv 2509.15221", "https://arxiv.org/abs/2509.15221")], role="跨 6 个操作系统的 GUI 下一步动作（带操作说明）")
E("scalecua-ground", dir="A", name="ScaleCUA grounding", hf=["OpenGVLab/ScaleCUA-Data"], new=True,
  files={"OpenGVLab/ScaleCUA-Data": [r"annotations/.*grounding.*\.jsonl"]}, modality="image", vtok=dict(image=SCREEN),
  role="跨平台 GUI 元素 grounding")
E("agenttrek", dir="E", also=["F"], name="AgentTrek", hf=["xlangai/AgentTrek"], new=True,
  files={"xlangai/AgentTrek": [r"agenttrek_1207\.json"]}, modality="text", role="由网络教程合成的 web agent 轨迹")

# ---- F Agentic
E("coderforge", dir="F", name="CoderForge-Preview", hf=["togethercomputer/CoderForge-Preview"], single=True, new=True,
  files={"togethercomputer/CoderForge-Preview": [r"trajectories/(R2E_Gym|SWE_Rebench|SWE_Smith)-.*\.parquet"]}, modality="text",
  note="filtered_reward1 是另外三个来源中 reward=1 的子集，不重复计；与 SWE-smith / R2E / SWE-rebench 轨迹题目重叠，三者择一",
  role="82 万条 OpenHands SWE 轨迹（R2E-Gym / SWE-rebench / SWE-smith 题目，带 reward）")
E("nemotron-terminal", dir="F", name="Nemotron-Terminal-Corpus", hf=["nvidia/Nemotron-Terminal-Corpus"], single=True, new=True,
  files={"nvidia/Nemotron-Terminal-Corpus": [r".*\.parquet"]}, modality="text",
  role="37 万条终端 agent 轨迹（分析 → 计划 → 批量 shell 命令 → 观察），含合成技能任务")
E("toolmind", dir="F", also=["C"], name="ToolMind", hf=["Nanbeige/ToolMind"], new=True,
  files={"Nanbeige/ToolMind": [r"graph_syn_datasets/.*\.jsonl"]}, modality="text",
  note="open_datasets/ 是 APIGen-MT 等公开集的重写版，不计", role="图合成的多轮工具调用，带 <think>")
E("nemotron-agentic-v1", dir="F", name="Nemotron-Agentic-v1", hf=["nvidia/Nemotron-Agentic-v1"], new=True, tier="dup",
  files={"nvidia/Nemotron-Agentic-v1": [r"data/.*\.jsonl"]}, modality="text", note="被 v2 / v4 覆盖，不重复计入",
  role="v1：interactive_agent + tool_calling")

# ---- G General
E("finevision", dir="G", name="FineVision", hf=[FV], single=True, priority=1, new=True,
  files={FV: [r"(?!" + FV_SKIP + r"/).*/train-.*\.parquet"]}, modality="image", vtok=dict(image=img_tok(1024, 1024)),
  links=[("博客", "https://huggingface.co/spaces/HuggingFaceM4/FineVision")],
  note="185 个子集统一格式并做过去重与质量打分；去掉了与 A / E 重复的 GUI 子集（aguvis-stage-1、screenqa 等）；card 标 9.5B answer token",
  role="2430 万样本、1730 万图：描述、OCR、文档、图表、数学、通用 VQA")
E("nemotron-vlm-v2", dir="G", name="Nemotron-VLM-Dataset-v2", hf=["nvidia/Nemotron-VLM-Dataset-v2"], single=True, priority=2, new=True,
  files={"nvidia/Nemotron-VLM-Dataset-v2": [r"[^/]+/[^/.][^/]*\.jsonl"]}, modality="image", vtok=dict(image=img_tok(800, 800)),
  note="部分子集图像需按脚本回源；含 CoT 与视频子集", role="815 万样本：OCR、图表、CoT 推理、视频理解")
E("llava-ov15", dir="G", name="LLaVA-OneVision-1.5-Instruct", hf=["mvp-lab/LLaVA-OneVision-1.5-Instruct-Data"], tier="ref", new=True,
  files={}, modality="image", note="2200 万样本，与 FineVision 有 46 个同名子集；作 FineVision 的替代来源，不重复计",
  role="LLaVA-OV-1.5 SFT 数据")
E("cauldron", dir="G", name="The Cauldron", hf=["HuggingFaceM4/the_cauldron"], tier="dup", new=True,
  files={}, modality="image", note="50 个子集中 48 个已包含在 FineVision", role="Idefics2 SFT 合集")
E("llava-video", dir="G", also=["B"], name="LLaVA-Video-178K", hf=["lmms-lab/LLaVA-Video-178K"], new=True,
  files={"lmms-lab/LLaVA-Video-178K": [r".*\.json"]}, modality="video", vtok=dict(video=vid_tok(640, 360)),
  role="17.8 万视频的稠密描述与开放 / 选择题 QA")
E("pixmo-docs", dir="G", name="PixMo-Docs", hf=["allenai/pixmo-docs"], new=True,
  files={"allenai/pixmo-docs": [r".*\.parquet"]}, modality="image", vtok=dict(image=img_tok(2000, 1400)),
  role="合成图表 / 表格 / 文档的 QA（25.5 万图）")


# Large multi-config parquet repos: rows from the /size API per config (tokens.estimate_stratified)
STRATA = {
    "eo-grounding": {EO: dict(group=r"([^/]+)/")}, "eo-state": {EO: dict(group=r"([^/]+)/")},
    "eo-afford": {EO: dict(group=r"([^/]+)/")}, "eo-physics": {EO: dict(group=r"([^/]+)/")},
    "eo-plan": {EO: dict(group=r"([^/]+)/")}, "eo-interleave": {EO: dict(group=r"([^/]+)/")},
    "robo2vlm": {"keplerccc/Robo2VLM-1": dict(group=r"data/(train)-", key="split", cfg="default")},
    "molmoact-mid": {"allenai/MolmoAct-Midtraining-Mixture": dict(group=r"([^/]+)/")},
    "molmoact-pre": {"allenai/MolmoAct-Pretraining-Mixture": dict(group=r"([^/]+)/")},
    "finevision": {FV: dict(group=r"([^/]+)/", one=True, rgs=2, workers=8)},
    "pixmo-docs": {"allenai/pixmo-docs": dict(group=r"([^/]+)/")},
    "action100m": {"facebook/action100m-preview": dict(group=r"(data)/", rows_cfg="default")},
    "coderforge": {"togethercomputer/CoderForge-Preview": dict(group=r"trajectories/([A-Za-z0-9_]+)-", key="split", cfg="trajectories")},
    "cwm-swe": {"marin-community/swe-rebench-v2-CodeWorldModeling": dict(group=r"(data)/", rows_cfg="default", splits=("traces",), picks=16)},
}
for _k, _v in STRATA.items():
    ENTRIES[_k]["strata"] = _v


# ---- per-entry fixes applied after token measurement
# Rows that always carry exactly one visual item (URL / id fields don't end in .jpg, so path detection undercounts)
PER_ROW = {"grit": dict(image=1), "pixmo-points": dict(image=1), "molmo2-videopoint": dict(video=1)}
for _k, _v in PER_ROW.items():
    ENTRIES[_k]["per_row"] = _v
# Video datasets whose "<image>" placeholder stands for the clip itself
for _k in ("llava-video", "egoit"):
    ENTRIES[_k]["count_images"] = False
# Order inside a direction: single-source first, then priority (lower first), then size.
# Embodied sources go ahead of web / code ones in D; tool-calling ahead of SWE in F.
for _k, _p in {"sharerobot-future": 1, "eo-physics": 1, "webworld": 2, "cwm-swe": 3, "codeio": 4, "py-state": 6,
               "toucan": 1, "nemotron-agentic-v2": 2, "nemotron-agentic-v4": 3, "nemotron-terminal": 3, "swe-zero": 4, "coderforge": 5,
               "cosmos-understanding": 1, "eo-state": 2, "robo2vlm": 3,
               "cosmos-reasoning": 1, "eo-plan": 2, "sharerobot-plan": 2, "aguvis-s2": 3,
               "sharerobot-afford": 3, "eo-afford": 3}.items():
    ENTRIES[_k]["priority"] = _p
ENTRIES["cwm-swe"]["note"] = ("CC-BY-4.0，另须遵守上游仓库许可。trace 长度极不均匀（单条最长约 87 万 token），"
                              "按行数估约 147B、按字节估约 223B，取前者；训练前必须按函数 / 测试切片")

ENTRIES["spar-7m"]["rgs"] = 64  # one 16M-row file sorted by qa_type: sample many small row groups


# ---- domain of each entry (shown on the page; lets each direction report how much is truly embodied)
DOMAIN = {}
for _k, _e in ENTRIES.items():
    hf = " ".join(_e["hf"]).lower()
    if "swe" in _k:
        DOMAIN[_k] = "code"
    elif any(s in _k for s in ("sharerobot", "cosmos", "eo-", "molmoact", "ecot", "robo", "aha", "vsi", "spar", "refspatial",
                             "embodied", "ebench", "era", "egoit", "egoplan", "action100m", "clevrer")):
        DOMAIN[_k] = "embodied"
    elif any(s in _k for s in ("aguvis", "os-atlas", "uground", "agentnet", "androidcontrol", "gui", "scalecua", "groundcua", "mobile-gui")):
        DOMAIN[_k] = "gui"
    elif any(s in _k for s in ("webworld", "wma", "webchain", "agenttrek", "openresearcher", "miroverse", "search", "asearcher", "webshaper")):
        DOMAIN[_k] = "web"
    elif any(s in _k for s in ("swe", "cwm", "codeio", "py-state", "coderforge", "terminal", "r2e")):
        DOMAIN[_k] = "code"
    elif _e["dir"] == "G" or _k in ("pixmo-points", "grit", "molmo2-videopoint"):
        DOMAIN[_k] = "general"
    else:
        DOMAIN[_k] = "tool"
for _k in ("bytesized32", "agenttraj-l"):
    DOMAIN[_k] = "text-env"
for _k, _d in DOMAIN.items():
    ENTRIES[_k]["domain"] = _d
DOMAIN_ZH = {"embodied": "具身 / 机器人", "gui": "GUI", "web": "Web", "code": "代码", "tool": "工具调用", "general": "通用视觉",
             "text-env": "文本环境"}

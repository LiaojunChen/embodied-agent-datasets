"""Curated dataset list, mirrored from 具身Agent训练数据集调研与配比建议.md.

Each entry: category, name, hf (list of HF dataset IDs; stats are summed),
links (list of (label, url)), scale, ability, stages (S0-S3), note.
"""

CATEGORIES = [
    ("embodied-failure", "具身 · 失败分析"),
    ("embodied-decision", "具身 · 多轮决策与反思"),
    ("embodied-reasoning", "具身 · 规划与物理推理"),
    ("embodied-spatial", "具身 · 空间定位"),
    ("general-agent", "通用 Agent"),
    ("tool", "工具调用"),
    ("swe", "SWE 编码 Agent"),
    ("web", "Web 搜索 / Deep Research"),
    ("gui", "GUI / 电脑操作"),
]


def arxiv(i):
    return ("arXiv " + i, "https://arxiv.org/abs/" + i)


def gh(repo):
    return ("GitHub", "https://github.com/" + repo)


DATASETS = [
    # 具身 · 失败分析
    dict(cat="embodied-failure", name="RoboFAC", hf=["MINT-SJTU/RoboFAC-dataset"], links=[arxiv("2505.12224")],
         scale="9,440 失败轨迹 / 78,623 QA", ability="失败检测、定位、归因、纠正", stages=["S2"], note="失败分析首选"),
    dict(cat="embodied-failure", name="AHA", hf=["ACIDE/AHA_Dataset_2e", "ACIDE/AHA_Dataset_2e_1p", "ACIDE/AHA_Dataset_2e_2p"],
         links=[arxiv("2410.00371"), ("项目页", "https://robofail-vlm.github.io/")],
         scale="RLBench 程序化失败注入", ability="失败检测 + 自由文本解释", stages=["S2"], note="数据规模小，主要借鉴 FailGen 方法"),
    dict(cat="embodied-failure", name="REFLECT", hf=[], links=[arxiv("2306.15724")],
         scale="方法参考", ability="经验摘要 → 失败解释 → 纠正规划", stages=[], note="无 HF 数据，借鉴方法"),
    # 具身 · 多轮决策与反思
    dict(cat="embodied-decision", name="Embodied-Reasoner", hf=["zwq2018/embodied_reasoner"], links=[arxiv("2503.21696")],
         scale="AI2-THOR 交互式搜索轨迹", ability="观察→思考→行动→反思的长链推理与自我纠错", stages=["S1", "S2"],
         note="公开数据中最接近“具身 + 反思重试”"),
    dict(cat="embodied-decision", name="EmbodiedBench 轨迹",
         hf=["EmbodiedBench/EB-Man_trajectory_dataset", "EmbodiedBench/EB-Alfred_trajectory_dataset", "EmbodiedBench/EB-Nav_trajectory_dataset"],
         links=[gh("EmbodiedBench/EmbodiedBench")], scale="多模型多任务轨迹", ability="高/低层具身多轮规划",
         stages=["S1", "S2"], note="基准配套数据，作补充"),
    dict(cat="embodied-decision", name="ERA 先验数据",
         hf=["EmbodiedReasoningAgent/EB-ALFRED_trajectory_augmented_prior_dataset", "EmbodiedReasoningAgent/EB-ALFRED_environment_anchored_prior_dataset",
             "EmbodiedReasoningAgent/EB-ALFRED_external_knowledge_prior_dataset", "EmbodiedReasoningAgent/EB-Man_trajectory_augmented_prior_dataset",
             "EmbodiedReasoningAgent/EB-Man_environment_anchored_prior_dataset", "EmbodiedReasoningAgent/EB-Man_external_knowledge_prior_dataset"],
         links=[gh("Embodied-Reasoning-Agent/Embodied-Reasoning-Agent")], scale="轨迹增强 / 环境锚定 / 外部知识三类先验",
         ability="小 VLM 具身 agent 冷启动", stages=["S1", "S2"], note="作补充"),
    # 具身 · 规划与物理推理
    dict(cat="embodied-reasoning", name="ShareRobot (RoboBrain)", hf=["BAAI/ShareRobot"], links=[arxiv("2502.21257")],
         scale="基于 OXE 的规划 / affordance / 轨迹标注", ability="任务分解、可操作区域、末端轨迹", stages=["S0", "S1"]),
    dict(cat="embodied-reasoning", name="Cosmos-Reason1 SFT", hf=["nvidia/Cosmos-Reason1-SFT-Dataset"], links=[arxiv("2503.15558")],
         scale="物理常识 + 具身推理 CoT", ability="物理常识、下一步动作推理", stages=["S0", "S1"]),
    dict(cat="embodied-reasoning", name="RoboVQA", hf=["Tianli/robovqa"], links=[arxiv("2311.00899"), gh("google-deepmind/robovqa")],
         scale="约 80 万 (视频, 文本) 对", ability="长程规划、成功检测", stages=["S0", "S1"],
         note="HF 为社区转换版，统计仅供参考；以官方 GitHub 为准"),
    dict(cat="embodied-reasoning", name="EgoPlan-IT / EgoPlan-Bench2", hf=["qiulu66/EgoPlan-Bench2"],
         links=[arxiv("2312.06722"), arxiv("2412.04447"), gh("ChenYi99/EgoPlan")],
         scale="第一视角视频 + 下一步规划", ability="第一视角任务进度理解与规划", stages=["S1"],
         note="HF 统计为评测集；训练集 EgoPlan-IT 见 GitHub"),
    # 具身 · 空间定位
    dict(cat="embodied-spatial", name="RefSpatial (RoboRefer)", hf=["JingkunAn/RefSpatial"], links=[arxiv("2506.04308")],
         scale="多步空间指代，2D/3D/仿真", ability="空间指代、输出点坐标", stages=["S1", "S2"]),
    dict(cat="embodied-spatial", name="RoboPoint", hf=["wentao-yuan/robopoint-data"], links=[arxiv("2406.10721")],
         scale="空间 affordance 点预测", ability="放置点 / 可操作点", stages=["S1", "S2"]),
    dict(cat="embodied-spatial", name="PixMo-Points", hf=["allenai/pixmo-points"], links=[arxiv("2409.17146")],
         scale="指点与计数标注", ability="通用 pointing", stages=["S1"]),
    dict(cat="embodied-spatial", name="VSI-590K (Cambrian-S)", hf=["nyu-visionx/VSI-590K"], links=[arxiv("2511.04670")],
         scale="59 万视频空间 QA", ability="距离、尺寸、相对方位、路线", stages=["S1"]),
    # 通用 Agent
    dict(cat="general-agent", name="ADP (Agent Data Protocol)", hf=["neulab/agent-data-collection"], links=[arxiv("2510.24702")],
         scale="18 个子集统一格式", ability="多源 agent 轨迹一站式获取", stages=["S1"]),
    dict(cat="general-agent", name="Nemotron-SFT-Agentic", hf=["nvidia/Nemotron-SFT-Agentic-v2", "nvidia/Nemotron-SFT-Agentic-v4"], links=[],
         scale="工业级，tool_calling / knowledge_agent", ability="多轮 / 并行工具调用", stages=["S1"], note="v2 为主，v4 为 2026-10 新版"),
    dict(cat="general-agent", name="Nemotron-RL-Agentic-FC", hf=["nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1"], links=[],
         scale="RL 任务集", ability="工具调用 RL", stages=["S3"]),
    dict(cat="general-agent", name="AgentGym", hf=["AgentGym/AgentTraj-L", "AgentGym/AgentEval"], links=[arxiv("2406.04151"), arxiv("2509.08755")],
         scale="14 个环境轨迹 + RL 框架", ability="多环境多轮决策", stages=["S1", "S3"]),
    dict(cat="general-agent", name="AgentInstruct", hf=["zai-org/AgentInstruct"], links=[], scale="数千条",
         ability="早期 agent 能力对齐", stages=["S1"], note="少量多样性补充"),
    dict(cat="general-agent", name="Agent-FLAN", hf=["internlm/Agent-FLAN"], links=[], scale="数万条",
         ability="格式 / 推理解耦 + 负样本", stages=["S1"], note="少量多样性补充"),
    # 工具调用
    dict(cat="tool", name="TOUCAN", hf=["Agent-Ark/Toucan-1.5M"], links=[("OpenReview", "https://openreview.net/pdf/a4766e6a0f6ac0295127649b4be1d9dcbb015746.pdf")],
         scale="150 万轨迹，~500 MCP server", ability="大规模真实工具使用", stages=["S0", "S1"]),
    dict(cat="tool", name="ToolBench", hf=["Yhyu13/ToolBench_toolllama_G123_dfs"], links=[gh("OpenBMB/ToolBench")],
         scale="12 万+，16k RapidAPI", ability="DFS 搜索树，含失败回溯", stages=["S1"], note="HF 为社区镜像，建议从官方取数据"),
    dict(cat="tool", name="xLAM-60k (APIGen)", hf=["Salesforce/xlam-function-calling-60k"], links=[],
         scale="6 万", ability="高质量函数调用格式", stages=["S1"]),
    dict(cat="tool", name="APIGen-MT", hf=["Salesforce/APIGen-MT-5k"], links=[arxiv("2504.03601")],
         scale="5k", ability="多轮 agent-用户交互", stages=["S1"]),
    dict(cat="tool", name="ToolACE", hf=["Team-ACE/ToolACE"], links=[], scale="约 1 万+", ability="API 多样性、并行与依赖调用", stages=["S1"]),
    dict(cat="tool", name="When2Call", hf=["nvidia/When2Call"], links=[arxiv("2504.18851")],
         scale="多选 + 偏好数据", ability="何时不该调用工具：追问、拒答", stages=["S1"]),
    dict(cat="tool", name="Hermes Function Calling", hf=["NousResearch/hermes-function-calling-v1"], links=[],
         scale="约 1 万+", ability="函数调用 / JSON 模式", stages=["S1"]),
    # SWE
    dict(cat="swe", name="SWE-smith 轨迹", hf=["SWE-bench/SWE-smith-trajectories"], links=[], scale="数万",
         ability="报错→修复循环", stages=["S0", "S1"]),
    dict(cat="swe", name="SWE-Gym", hf=["SWE-Gym/SWE-Gym"], links=[], scale="2.4k 可执行实例", ability="SWE RL 环境", stages=["S3"]),
    dict(cat="swe", name="R2E-Gym", hf=["R2E-Gym/R2EGym-SFT-Trajectories", "R2E-Gym/R2EGym-Verifier-Trajectories"], links=[],
         scale="SFT + Verifier 轨迹", ability="轨迹 + 验证器，可训 critic", stages=["S1"]),
    dict(cat="swe", name="SWE-rebench", hf=["nebius/SWE-rebench-V2", "nebius/SWE-rebench-openhands-trajectories"], links=[],
         scale="持续更新", ability="防污染 SWE 数据", stages=["S1"]),
    dict(cat="swe", name="Nebius SWE-agent 轨迹", hf=["nebius/SWE-agent-trajectories"], links=[], scale="约 8 万",
         ability="成功 + 失败轨迹均有", stages=["S0", "S1"]),
    dict(cat="swe", name="NVIDIA SWE-Zero / SWE-Hero", hf=["nvidia/SWE-Zero-openhands-trajectories", "nvidia/SWE-Hero-openhands-trajectories"],
         links=[], scale="OpenHands 轨迹", ability="免执行 / 基于执行两种 SFT 轨迹", stages=["S1"]),
    dict(cat="swe", name="SWE RL 任务池", hf=["SWE-bench/SWE-smith", "R2E-Gym/R2E-Gym-Subset", "nebius/SWE-rebench"], links=[],
         scale="数千～数万可执行实例", ability="SWE RL 任务采样", stages=["S3"]),
    dict(cat="swe", name="Scale-SWE", hf=[], links=[arxiv("2602.09892")], scale="6M PR / 5.2k 仓库", ability="大规模可执行 SWE 环境",
         stages=["S0"], note="仅论文，未发现 HF 数据"),
    dict(cat="swe", name="Open-SWE-Traces", hf=[], links=[arxiv("2606.16038")], scale="多语言轨迹", ability="thinking / non-thinking 蒸馏",
         stages=["S1"], note="仅论文"),
    # Web
    dict(cat="web", name="Search-R1 训练集", hf=["PeterJinGo/nq_hotpotqa_train"], links=[arxiv("2503.09516")],
         scale="NQ + HotpotQA", ability="搜索 RL 起点", stages=["S3"]),
    dict(cat="web", name="ASearcher", hf=["inclusionAI/ASearcher-train-data"], links=[arxiv("2508.07976")],
         scale="合成高难 QA", ability="长程（数十轮）搜索 RL", stages=["S1", "S3"]),
    dict(cat="web", name="MiroVerse", hf=["miromind-ai/MiroVerse-v0.1"], links=[], scale="deep research 训练数据",
         ability="长程搜索与综合", stages=["S1"]),
    dict(cat="web", name="OpenResearcher", hf=["OpenResearcher/OpenResearcher-Dataset"], links=[gh("TIGER-AI-Lab/OpenResearcher")],
         scale="长程 research 轨迹", ability="长程搜索与综合", stages=["S1"]),
    dict(cat="web", name="WebShaper", hf=["Alibaba-NLP/WebShaper"], links=[arxiv("2507.15061")], scale="合成 IS 任务",
         ability="信息检索 agent", stages=["S1", "S3"], note="作补充"),
    dict(cat="web", name="WebChain", hf=["webagentlab/webchain"], links=[arxiv("2603.05295")], scale="31,725 人工轨迹",
         ability="真实网站交互", stages=["S1"], note="新发布，作补充"),
    # GUI
    dict(cat="gui", name="Aguvis", hf=["xlangai/aguvis-stage1", "xlangai/aguvis-stage2"], links=[arxiv("2412.04454")],
         scale="整合 AITW / AMEX / AndroidControl 等", ability="GUI grounding + 规划", stages=["S1"], note="最常用的 GUI SFT 混合包"),
    dict(cat="gui", name="AgentNet (OpenCUA)", hf=["xlangai/AgentNet"], links=[], scale="大规模桌面轨迹",
         ability="带反思 CoT 的电脑操作", stages=["S1"]),
    dict(cat="gui", name="GUI-Odyssey", hf=["OpenGVLab/GUI-Odyssey"], links=[], scale="跨 App 长程任务",
         ability="视觉观测 + 动作序列", stages=["S1"]),
    dict(cat="gui", name="AndroidControl", hf=["ckg/AndroidControlParsedWithImages-20k"],
         links=[("官方数据", "https://github.com/google-research/google-research/tree/master/android_control")],
         scale="约 1.5 万任务", ability="高 / 低层指令双标注", stages=["S1"], note="HF 为社区带图解析版"),
    dict(cat="gui", name="OS-Atlas", hf=["OS-Copilot/OS-Atlas-data"], links=[arxiv("2410.23218")], scale="百万级元素",
         ability="GUI 元素 grounding", stages=["S1"]),
    dict(cat="gui", name="UGround", hf=["osunlp/UGround-V1-Data"], links=[arxiv("2410.05243")], scale="百万级元素",
         ability="GUI 元素 grounding", stages=["S1"]),
]

STAGES = [
    dict(id="S0", title="S0 Mid-training（可选）", goal="熟悉工具调用格式和代码执行反馈", size="10B–50B tokens", mix=[
        ("通用预训练语料", 50), ("代码语料", 20), ("工具使用合成语料（TOUCAN 等）", 15),
        ("SWE 轨迹", 10), ("具身文本（Cosmos-Reason1 / ShareRobot / RoboVQA）", 5)]),
    dict(id="S1", title="S1 Agentic SFT", goal="打通多轮工具调用和多领域 agent 能力", size="1B–5B tokens", mix=[
        ("工具调用", 30), ("SWE / 代码 agent", 20), ("具身多轮与推理", 15), ("通用对话 / 推理", 15),
        ("Web / Deep Research", 10), ("GUI 操作", 10)]),
    dict(id="S2", title="S2 具身 + 反思 SFT", goal="迁移到具身，学会失败分析与重试", size="0.2B–1B tokens", mix=[
        ("自建 RoboTwin 工具调用轨迹（成功）", 30), ("自建 失败→归因→重试轨迹", 25), ("RoboFAC + AHA 失败分析 QA", 15),
        ("S1 数据回放", 15), ("空间定位 + EmbodiedBench / ERA", 10), ("Embodied-Reasoner 反思轨迹", 5)]),
    dict(id="S3", title="S3 多轮 RL", goal="提升实际成功率和自主重试能力", size="按环境步数计", mix=[
        ("RoboTwin 等具身仿真", 50), ("ALFWorld / EB-ALFRED", 15), ("τ-bench / Nemotron-RL-FC", 15),
        ("SWE-Gym / R2E-Gym / SWE-smith", 10), ("Web 搜索（Search-R1 / ASearcher）", 10)]),
]

EVALS = [
    ("工具调用", "BFCL v3/v4、τ-bench / τ²-bench、When2Call"),
    ("多轮代码 agent", "SWE-bench Verified、SWE-rebench、Terminal-Bench"),
    ("搜索 / Research", "GAIA、BrowseComp(-Plus)、WebWalkerQA"),
    ("GUI", "OSWorld(-Verified)、AndroidWorld、ScreenSpot-Pro"),
    ("具身规划", "EmbodiedBench、ALFWorld、Embodied Agent Interface、EgoPlan-Bench2"),
    ("具身推理 / 空间", "ERQA、RoboVQA、VSI-Bench、Where2Place / RefSpatial-Bench"),
    ("失败分析", "RoboFAC 测试集、AHA 测试集"),
    ("具身执行", "RoboTwin、LIBERO、CALVIN、VLN-CE"),
]

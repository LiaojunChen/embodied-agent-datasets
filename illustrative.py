"""Format illustrations for entries without publicly readable data.

These are NOT original records. The page labels them as 格式示意 and shows `why`.
"""

ILLUSTRATIVE = {
    "xLAM-60k (APIGen)": dict(
        why="gated 数据集，需在 HF 申请后才能读取；按 dataset card 的字段（query / tools / answers，后两者为 JSON 字符串）整理",
        record={
            "id": 0,
            "query": "Where can I find live giveaways for beta access and games?",
            "tools": [{"name": "live_giveaways_by_type",
                       "description": "Retrieve live giveaways from the GamerPower API based on the specified type.",
                       "parameters": {"type": {"description": "The type of giveaways to retrieve (e.g., game, loot, beta).",
                                               "type": "str", "default": "game"}}}],
            "answers": [{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
                        {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}],
        }),
    "MiroVerse": dict(
        why="gated 数据集，需在 HF 申请后才能读取；字段按 card 中的 SFT jsonl 配置推断，申请后需核对",
        record={
            "messages": [
                {"role": "system", "content": "You are a research agent. Use the tools to search and browse, then answer."},
                {"role": "user", "content": "Which university did the director of the 2010 film that won the Palme d'Or attend?"},
                {"role": "assistant", "content": "<think>先确认 2010 年金棕榈获奖影片。</think>",
                 "tool_calls": [{"name": "google_search", "arguments": {"q": "2010 Palme d'Or winner"}}]},
                {"role": "tool", "content": "Uncle Boonmee Who Can Recall His Past Lives — directed by Apichatpong Weerasethakul ..."},
                {"role": "assistant", "content": "<think>再查导演的教育背景。</think>",
                 "tool_calls": [{"name": "scrape_website", "arguments": {"url": "https://en.wikipedia.org/wiki/Apichatpong_Weerasethakul"}}]},
                {"role": "tool", "content": "... graduated from Khon Kaen University ... MFA, School of the Art Institute of Chicago ..."},
                {"role": "assistant", "content": "Khon Kaen University（本科），之后在 School of the Art Institute of Chicago 获 MFA。"},
            ]
        }),
    "UGround": dict(
        why="gated 数据集，需在 HF 申请后才能读取；按论文描述（截图 + 指代表达 → 归一化坐标）整理",
        record={
            "image": "<web screenshot>",
            "conversations": [
                {"from": "human", "value": "<image>\nIn the screenshot, where is the search button next to the address field?"},
                {"from": "gpt", "value": "(612, 87)"},
            ],
            "note": "坐标为 [0, 1000) 归一化像素坐标",
        }),
    "REFLECT": dict(
        why="论文未发布 HF 数据；按论文中的“分层经验摘要 → 失败解释 → 纠正规划”流程整理",
        record={
            "task": "Put the egg in the pot and boil it",
            "sensory_summary": [
                {"t": "00:12", "event": "pick_up(egg) succeeded", "audio": "none"},
                {"t": "00:25", "event": "put_in(egg, pot)", "audio": "cracking sound"},
                {"t": "00:31", "event": "egg not detected inside pot", "audio": "none"},
            ],
            "failure_explanation": "The egg was dropped from too high above the pot at 00:25 and cracked on the rim.",
            "correction_plan": ["pick_up(egg)", "move_above(pot, height=low)", "place_in(egg, pot)", "toggle_on(stove)"],
        }),
    "Scale-SWE": dict(
        why="论文未发布 HF 数据；按论文描述的三 agent 流水线产物（可执行环境 + 生成测试 + 问题描述）整理",
        record={
            "repo": "example-org/example-lib",
            "base_commit": "a1b2c3d",
            "environment": {"dockerfile": "FROM python:3.11 ...", "setup_commands": ["pip install -e .[test]"]},
            "problem_statement": "parse_config() raises KeyError when the optional 'extras' section is missing.",
            "generated_tests": {"FAIL_TO_PASS": ["tests/test_config.py::test_missing_extras"],
                                "PASS_TO_PASS": ["tests/test_config.py::test_basic"]},
            "gold_patch": "diff --git a/lib/config.py b/lib/config.py ...",
        }),
    "Open-SWE-Traces": dict(
        why="论文未发布 HF 数据；按论文描述（多语言、thinking / non-thinking 两种模式的 agent 轨迹）整理",
        record={
            "instance_id": "example__repo-1234",
            "language": "go",
            "mode": "thinking",
            "messages": [
                {"role": "user", "content": "<issue> HTTP client ignores the configured timeout </issue>"},
                {"role": "assistant", "content": "<think>先找 client 构造处。</think>",
                 "tool_calls": [{"name": "bash", "arguments": {"cmd": "grep -rn 'Timeout' pkg/client"}}]},
                {"role": "tool", "content": "pkg/client/client.go:42: c := &http.Client{}"},
                {"role": "assistant", "tool_calls": [{"name": "edit", "arguments": {"path": "pkg/client/client.go",
                                                                                      "patch": "c := &http.Client{Timeout: cfg.Timeout}"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "assistant", "tool_calls": [{"name": "bash", "arguments": {"cmd": "go test ./pkg/client/..."}}]},
                {"role": "tool", "content": "ok  example/pkg/client 0.41s"},
            ],
            "resolved": True,
        }),
}

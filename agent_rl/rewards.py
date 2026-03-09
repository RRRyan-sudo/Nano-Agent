"""
Agent Reward 函数
用于 GRPO 训练，评估模型的工具调用能力和任务完成度

Reward 组件:
1. tool_call_valid (0.3): 能否解析出合法的工具调用
2. tool_selection (0.2): 是否选择了正确的工具
3. task_completion (1.0): 执行工具后结果是否匹配预期
4. overlong_penalty (-0.5): 输出过长的惩罚
5. repetition_pen (-1.0): 严重重复的惩罚
"""
import re
import numpy as np

from config import AGENT_REWARD_WEIGHTS
from tools import parse_tool_calls, SafeToolExecutor


# 全局缓冲，供 WandB Callback 读取
_reward_components_buffer = {}


def normalize_answer(s: str) -> str:
    """标准化答案字符串：去除空白、引号等。"""
    s = s.strip()
    s = s.strip('"').strip("'")
    s = s.replace(",", "").replace(" ", "")
    return s


def verify_result(tool_results: list[dict], expected_answer: str) -> bool:
    """
    验证工具执行结果是否匹配预期答案。
    检查预期答案是否出现在任何工具结果中。
    """
    if not tool_results or not expected_answer:
        return False

    expected_norm = normalize_answer(expected_answer)

    for tr in tool_results:
        result_text = tr.get("result", "")
        if not result_text or result_text.startswith("[error]"):
            continue

        # 精确匹配（标准化后）
        result_norm = normalize_answer(result_text)
        if expected_norm == result_norm:
            return True

        # 检查预期答案是否出现在结果输出中
        # 按行检查，处理多行输出
        for line in result_text.strip().split("\n"):
            line_norm = normalize_answer(line)
            if expected_norm == line_norm:
                return True

        # 对于列表/集合类型的答案，做更宽松的匹配
        if expected_norm in normalize_answer(result_text):
            return True

    return False


def repetition_penalty(completion: str) -> float:
    """重复惩罚：词级别 unique ratio < 0.4 则触发。"""
    words = completion.split()
    if len(words) < 15:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.4:
        return AGENT_REWARD_WEIGHTS["repetition_pen"]
    return 0.0


def agent_reward_fn(
    completions: list[str],
    prompts: list[str] = None,
    **kwargs,
) -> list[float]:
    """
    GRPO reward 函数：评估单轮工具调用的质量。

    接收一批 completions，返回对应 reward 列表。
    kwargs 中包含 dataset 的其他字段（expected_answer, expected_tool, task_type）。
    """
    expected_answers = kwargs.get("expected_answer", [None] * len(completions))
    expected_tools = kwargs.get("expected_tool", [None] * len(completions))

    rewards = []
    valid_scores = []
    select_scores = []
    complete_scores = []
    rep_scores = []

    for i, completion in enumerate(completions):
        r = 0.0
        expected_answer = expected_answers[i] if i < len(expected_answers) else None
        expected_tool = expected_tools[i] if i < len(expected_tools) else None

        # 1. 工具调用合法性
        tool_calls = parse_tool_calls(completion)
        valid_score = AGENT_REWARD_WEIGHTS["tool_call_valid"] if tool_calls else 0.0
        r += valid_score

        # 2. 工具选择正确性
        select_score = 0.0
        if tool_calls and expected_tool:
            if any(tc["name"] == expected_tool for tc in tool_calls):
                select_score = AGENT_REWARD_WEIGHTS["tool_selection"]
        r += select_score

        # 3. 任务完成度（执行工具并验证结果）
        complete_score = 0.0
        if tool_calls and expected_answer:
            executor = SafeToolExecutor(timeout=5)
            try:
                results = []
                for tc in tool_calls:
                    result = executor.execute(tc["name"], tc["arguments"])
                    results.append({"name": tc["name"], "result": result})
                if verify_result(results, expected_answer):
                    complete_score = AGENT_REWARD_WEIGHTS["task_completion"]
            finally:
                executor.cleanup()
        r += complete_score

        # 4. 超长惩罚
        if len(completion.split()) > 300:
            r += AGENT_REWARD_WEIGHTS["overlong_penalty"]

        # 5. 重复惩罚
        rep_score = repetition_penalty(completion)
        r += rep_score

        rewards.append(r)
        valid_scores.append(valid_score)
        select_scores.append(select_score)
        complete_scores.append(complete_score)
        rep_scores.append(rep_score)

    # 写入缓冲，WandB Callback 读取
    _reward_components_buffer.update({
        "reward/tool_valid":    float(np.mean(valid_scores)),
        "reward/tool_select":   float(np.mean(select_scores)),
        "reward/task_complete": float(np.mean(complete_scores)),
        "reward/rep_penalty":   float(np.mean(rep_scores)),
        "reward/mean":          float(np.mean(rewards)),
        "reward/std":           float(np.std(rewards)),
        "reward/completion_rate": float(np.mean([c > 0 for c in complete_scores])),
    })

    return rewards

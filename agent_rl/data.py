"""
数据加载与格式化
- Agent SFT 数据：glaive-function-calling-v2 → Qwen ChatML 格式
- Agent RL 任务：可验证的代码执行任务，用于 GRPO 训练
"""
import json
import re
from datasets import load_dataset, Dataset

from config import AGENT_SFT_CONFIG, AGENT_SYSTEM_PROMPT
from tools import TOOL_SCHEMAS


# ============================================================
# SFT 数据加载
# ============================================================

def _convert_glaive_to_messages(example: dict) -> dict | None:
    """
    将 glaive-function-calling-v2 的格式转换为 Qwen ChatML messages。

    glaive 原始格式:
      system_prompt: "SYSTEM: You are a helpful assistant..."
      chat: "USER: ...\nASSISTANT: ...\n<|endoftext|>"
      （或含有 FUNCTION RESPONSE 等）
    """
    chat_text = example.get("chat", "")
    if not chat_text:
        return None

    messages = []

    # 添加我们的系统提示（含工具定义）
    messages.append({
        "role": "system",
        "content": AGENT_SYSTEM_PROMPT,
    })

    # 解析对话轮次
    # glaive 格式: USER: ... \n ASSISTANT: ... \n FUNCTION RESPONSE: ...
    parts = re.split(r'\n(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)', chat_text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("USER:"):
            content = part[len("USER:"):].strip()
            if content:
                messages.append({"role": "user", "content": content})

        elif part.startswith("ASSISTANT:"):
            content = part[len("ASSISTANT:"):].strip()
            # 移除 <|endoftext|> 标记
            content = content.replace("<|endoftext|>", "").strip()
            if content:
                messages.append({"role": "assistant", "content": content})

        elif part.startswith("FUNCTION RESPONSE:"):
            content = part[len("FUNCTION RESPONSE:"):].strip()
            if content:
                messages.append({"role": "tool", "content": content})

    # 过滤：至少要有 user + assistant 两轮
    user_count = sum(1 for m in messages if m["role"] == "user")
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    if user_count < 1 or assistant_count < 1:
        return None

    return {"messages": messages}


def load_agent_sft_data(tokenizer, max_samples: int = None):
    """
    加载 glaive-function-calling-v2 数据集，转为 ChatML 格式用于 SFT。

    返回 Dataset，每条样本包含 "text" 字段（apply_chat_template 后的完整对话）。
    """
    if max_samples is None:
        max_samples = AGENT_SFT_CONFIG["max_train_samples"]

    print(f"Loading glaive-function-calling-v2 (max {max_samples} samples)...")
    dataset = load_dataset(AGENT_SFT_CONFIG["dataset_name"], split="train")

    # 随机选取子集
    if len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    # 转换格式
    converted = []
    for example in dataset:
        result = _convert_glaive_to_messages(example)
        if result is not None:
            # 用 tokenizer 的 chat template 格式化
            try:
                text = tokenizer.apply_chat_template(
                    result["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                converted.append({"text": text})
            except Exception:
                continue

    sft_dataset = Dataset.from_list(converted)
    print(f"Agent SFT dataset size: {len(sft_dataset)}")
    return sft_dataset


# ============================================================
# RL 任务数据
# ============================================================

# 可验证的代码执行任务集
# 每个任务：prompt 描述问题，expected_answer 是预期输出，expected_tool 是应使用的工具
AGENT_RL_TASKS = [
    # ── 基础计算 ──
    {
        "task": "Calculate the sum of all prime numbers less than 50.",
        "expected_answer": "328",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate the factorial of 12.",
        "expected_answer": "479001600",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "What is the 20th Fibonacci number?",
        "expected_answer": "6765",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate 2^32 - 1.",
        "expected_answer": "4294967295",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "What is the sum of digits of 999999999999?",
        "expected_answer": "108",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate the greatest common divisor of 462 and 1071.",
        "expected_answer": "21",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "How many even numbers are there between 1 and 100 (inclusive)?",
        "expected_answer": "50",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "What is the sum of the first 100 positive integers?",
        "expected_answer": "5050",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate the number of permutations of 5 items taken 3 at a time (P(5,3)).",
        "expected_answer": "60",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "What is the least common multiple of 12 and 18?",
        "expected_answer": "36",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    # ── 字符串处理 ──
    {
        "task": "Count the number of vowels in the string 'reinforcement learning is fascinating'.",
        "expected_answer": "12",
        "expected_tool": "python_exec",
        "task_type": "string_code",
    },
    {
        "task": "Reverse the string 'hello world' and output the result.",
        "expected_answer": "dlrow olleh",
        "expected_tool": "python_exec",
        "task_type": "string_code",
    },
    {
        "task": "Count the number of words in 'the quick brown fox jumps over the lazy dog'.",
        "expected_answer": "9",
        "expected_tool": "python_exec",
        "task_type": "string_code",
    },
    {
        "task": "Find all unique characters in 'abracadabra' and output them sorted alphabetically, joined by commas.",
        "expected_answer": "a,b,c,d,r",
        "expected_tool": "python_exec",
        "task_type": "string_code",
    },
    {
        "task": "Convert the string 'Hello World' to lowercase and replace spaces with underscores.",
        "expected_answer": "hello_world",
        "expected_tool": "python_exec",
        "task_type": "string_code",
    },
    # ── 算法题 ──
    {
        "task": "Write a function to check if 'racecar' is a palindrome and print True or False.",
        "expected_answer": "True",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Write a function to check if 'hello' is a palindrome and print True or False.",
        "expected_answer": "False",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Sort the list [5, 3, 8, 1, 9, 2, 7] in ascending order and print the result.",
        "expected_answer": "[1, 2, 3, 5, 7, 8, 9]",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Find the second largest number in [15, 3, 22, 7, 19, 22, 8] and print it.",
        "expected_answer": "19",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Count the number of prime numbers between 1 and 100 (inclusive).",
        "expected_answer": "25",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Compute the sum of squares of the first 10 positive integers.",
        "expected_answer": "385",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Find the maximum subarray sum of [-2, 1, -3, 4, -1, 2, 1, -5, 4] using Kadane's algorithm.",
        "expected_answer": "6",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Convert the decimal number 255 to binary and print it (without '0b' prefix).",
        "expected_answer": "11111111",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Calculate the mean of [10, 20, 30, 40, 50] and print it as a float.",
        "expected_answer": "30.0",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Find all pairs of numbers in [1, 2, 3, 4, 5, 6] that sum to 7, print the count.",
        "expected_answer": "3",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    # ── 数据处理 ──
    {
        "task": "Create a dictionary mapping each character in 'hello' to its frequency, print the frequency of 'l'.",
        "expected_answer": "2",
        "expected_tool": "python_exec",
        "task_type": "data_code",
    },
    {
        "task": "Flatten the nested list [[1, 2], [3, 4], [5, 6]] into a single list and print it.",
        "expected_answer": "[1, 2, 3, 4, 5, 6]",
        "expected_tool": "python_exec",
        "task_type": "data_code",
    },
    {
        "task": "Given the list [1, 2, 2, 3, 3, 3, 4], remove duplicates and print the sorted result.",
        "expected_answer": "[1, 2, 3, 4]",
        "expected_tool": "python_exec",
        "task_type": "data_code",
    },
    {
        "task": "Merge two sorted lists [1, 3, 5, 7] and [2, 4, 6, 8] into one sorted list and print it.",
        "expected_answer": "[1, 2, 3, 4, 5, 6, 7, 8]",
        "expected_tool": "python_exec",
        "task_type": "data_code",
    },
    {
        "task": "Generate a list of squares from 1 to 10 and print the sum.",
        "expected_answer": "385",
        "expected_tool": "python_exec",
        "task_type": "data_code",
    },
    # ── 更多计算题（增加多样性）──
    {
        "task": "What is 123456789 * 987654321?",
        "expected_answer": "121932631112635269",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate the value of pi to 10 decimal places using Python's math module.",
        "expected_answer": "3.1415926536",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "How many digits does 2^1000 have?",
        "expected_answer": "302",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate the sum of all multiples of 3 or 5 below 1000.",
        "expected_answer": "233168",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "What is the 100th prime number?",
        "expected_answer": "541",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Compute the square root of 2 to 8 decimal places.",
        "expected_answer": "1.41421356",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "What is the sum of all even Fibonacci numbers up to 4000000?",
        "expected_answer": "4613732",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate C(10, 4), the number of combinations of 10 choose 4.",
        "expected_answer": "210",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "What is the product of all single-digit prime numbers (2, 3, 5, 7)?",
        "expected_answer": "210",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate the sum of cubes of numbers from 1 to 5.",
        "expected_answer": "225",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    # ── 更多字符串/算法题 ──
    {
        "task": "Count the number of uppercase letters in 'Hello World, This Is A Test'.",
        "expected_answer": "6",
        "expected_tool": "python_exec",
        "task_type": "string_code",
    },
    {
        "task": "Find the longest word in the sentence 'the quick brown fox jumps over the lazy dog' and print it.",
        "expected_answer": "quick",
        "expected_tool": "python_exec",
        "task_type": "string_code",
    },
    {
        "task": "Check if the string 'listen' is an anagram of 'silent'. Print True or False.",
        "expected_answer": "True",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Compute the Hamming distance between the strings 'karolin' and 'kathrin'.",
        "expected_answer": "3",
        "expected_tool": "python_exec",
        "task_type": "algo_code",
    },
    {
        "task": "Find the median of the list [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5] and print it.",
        "expected_answer": "4",
        "expected_tool": "python_exec",
        "task_type": "data_code",
    },
    {
        "task": "What is the binary representation of the number 42 (without '0b' prefix)?",
        "expected_answer": "101010",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Calculate the sum of ASCII values of all characters in the string 'Python'.",
        "expected_answer": "642",
        "expected_tool": "python_exec",
        "task_type": "string_code",
    },
    {
        "task": "How many integers from 1 to 1000 are divisible by either 7 or 11 but not both?",
        "expected_answer": "220",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Find the number of trailing zeros in 100! (100 factorial).",
        "expected_answer": "24",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
    {
        "task": "Generate the first 10 triangular numbers and print them as a list.",
        "expected_answer": "[1, 3, 6, 10, 15, 21, 28, 36, 45, 55]",
        "expected_tool": "python_exec",
        "task_type": "math_code",
    },
]


def load_agent_rl_tasks(tokenizer) -> Dataset:
    """
    加载 Agent RL 任务数据集。
    每条样本包含:
      - prompt: 格式化的对话文本（system + user）
      - expected_answer: 预期输出字符串
      - expected_tool: 应使用的工具名
      - task_type: 任务类别
    """
    print(f"Loading {len(AGENT_RL_TASKS)} agent RL tasks...")

    processed = []
    for task in AGENT_RL_TASKS:
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": task["task"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processed.append({
            "prompt": prompt,
            "expected_answer": task["expected_answer"],
            "expected_tool": task["expected_tool"],
            "task_type": task["task_type"],
        })

    dataset = Dataset.from_list(processed)
    print(f"Agent RL task dataset size: {len(dataset)}")
    return dataset


def load_agent_eval_tasks() -> list[dict]:
    """加载评测用任务（直接返回原始列表，evaluate.py 使用）。"""
    return AGENT_RL_TASKS

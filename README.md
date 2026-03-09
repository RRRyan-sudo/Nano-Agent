# Agent Zero to Hero

从零搭建 AI Agent，再到用强化学习训练出会用工具的模型。本仓库按学习难度递进，包含三个独立项目。

```
agent_zero2hero/
├── simple_agent/        # 项目一：从零实现 AI Agent
├── simple_posttrain/    # 项目二：LLM 后训练入门（SFT + GRPO）
└── agent_rl/            # 项目三：Agent 专项后训练（工具调用能力强化）
```

---

## 项目一：simple_agent — 从零实现 AI Agent

### 目标

理解 Agent 的核心机制：**LLM 推理 → 工具调用 → 观察结果 → 继续推理** 的循环。

### 项目结构

```
simple_agent/
├── zero2agent.py   # 从零手写的 Agent Loop（核心实现）
├── agent_cli.py    # 命令行入口，调用 zero2agent
└── react_agent.py  # 基于 LangGraph 的 ReAct Agent（进阶版）
```

### 算法：Agent Loop / ReAct

**zero2agent.py** 用最简洁的方式实现了 Agent 的核心循环：

```
while True:
    response = LLM(messages + tools)
    if response.tool_calls:
        result = execute_tool(tool_calls)
        messages.append(result)   # 把工具结果喂回 LLM
    else:
        return response.content   # LLM 直接回答，结束
```

内置 4 个工具：

| 工具 | 功能 |
|------|------|
| `shell_exec` | 执行 Shell 命令 |
| `file_read` | 读取文件内容 |
| `file_write` | 写入文件 |
| `python_exec` | 在子进程中运行 Python 代码 |

**react_agent.py** 用 LangGraph 实现了同样的逻辑，将 Agent 建模为状态图（StateGraph），节点为 `agent`（LLM 推理）和 `tools`（工具执行），通过条件边路由，并集成 `MemorySaver` 实现多轮记忆。

### 使用方法

前提：本地启动 vLLM 服务（兼容 OpenAI API）：

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
```

**方式 1：手写 Agent Loop**

```bash
cd simple_agent
python agent_cli.py
```

```
You> 帮我写一个快速排序，存到 /tmp/sort.py，然后运行它验证
  [tool] python_exec({"code": "..."})
  [result] ...
Agent> 已完成，快速排序已写入并验证通过。
```

**方式 2：LangGraph ReAct Agent**

```bash
pip install langgraph langchain-openai python-dotenv
python react_agent.py
# 输入 'new' 开启新对话，'quit' 退出
```

---

## 项目二：simple_posttrain — LLM 后训练入门

### 目标

以 Qwen2.5-1.5B 为例，走通 **SFT → GRPO** 完整后训练流程，验证 RL 能提升数学推理能力。

### 项目结构

```
simple_posttrain/
├── config.py        # 统一配置（模型、路径、超参、Reward 权重）
├── sft_train.py     # Stage 1：监督微调
├── grpo_train.py    # Stage 2：GRPO 强化学习
├── evaluate.py      # 评测脚本
└── requirements.txt
```

### 训练流程

```
Qwen2.5-1.5B (Base)
      │
      ▼  Stage 1: SFT
  SFT LoRA 微调（ultrachat_200k，20k 条多轮对话）
      │  merge LoRA → outputs/sft_merged/
      ▼  Stage 2: GRPO
  GRPO 强化学习（GSM8K 小学数学题）
      │
      ▼
  outputs/grpo/final/
```

### 算法

**Stage 1 — SFT（监督微调）**

- 数据集：`HuggingFaceH4/ultrachat_200k`（通用多轮对话，20k 条）
- 方法：LoRA（rank=64）+ sequence packing，使用 `trl.SFTTrainer`
- 目标：让 Base 模型学会对话格式，为 RL 阶段提供稳定起点

**Stage 2 — GRPO（Group Relative Policy Optimization）**

GRPO 是 DeepSeek-R1 使用的 RL 算法，核心思想：对同一个 prompt 并行生成 N 个回答，以**组内相对奖励**替代绝对奖励（无需 Critic 网络），用 KL 散度约束防止策略偏离参考模型太远。

- 数据集：`openai/gsm8k`（2400 条数学应用题）
- 每个 prompt 生成 4 个回答（`num_generations=4`），组内归一化 reward
- KL penalty 系数 `beta=0.1`

Reward 函数由 4 个分量组成：

| 分量 | 权重 | 说明 |
|------|------|------|
| `answer_correct` | +1.0 | 答案数字与 ground truth 完全匹配 |
| `format_complete` | +0.5 | 同时包含 `<reasoning>` 和 `<answer>` 标签 |
| `step_quality` | +0.3 | `<reasoning>` 内容长度适中（50-400 字）|
| `repetition_pen` | -1.0 | 词级别 unique ratio < 0.4 时惩罚 |

模型被训练输出如下格式：

```
<reasoning>
Step 1: ...
Step 2: ...
</reasoning>
<answer>42</answer>
```

### 使用方法

```bash
cd simple_posttrain
pip install -r requirements.txt

# Stage 1: SFT（约 2-3 小时，单卡 A100）
python sft_train.py

# Stage 2: GRPO（约 1 小时）
python grpo_train.py

# 评测
python evaluate.py
```

训练过程通过 WandB 监控，关键指标：

- `reward/answer_score`：答案准确率（最直接的改善信号）
- `reward/format_score`：格式合规率
- `train/kl`：KL 散度（不应持续增大）
- `eval/accuracy`：定期评测准确率

---

## 项目三：agent_rl — Agent 专项后训练

### 目标

在项目二的基础上，针对 **工具调用能力** 做专项强化。训练模型不仅能生成正确的工具调用格式，还能通过工具执行真正完成任务。

### 项目结构

```
agent_rl/
├── config.py           # 配置（模型、超参、Reward 权重、System Prompt）
├── data.py             # SFT 数据加载 + 50 个可验证 RL 任务
├── tools.py            # 工具实现 + SafeToolExecutor + 工具调用解析
├── rewards.py          # Agent Reward 函数
├── sft_train.py        # Stage 1：工具调用 SFT 冷启动
├── grpo_train.py       # Stage 2：单轮 Agent GRPO
├── multi_turn_grpo.py  # Stage 2+：多轮 Agent GRPO
├── evaluate.py         # 评测脚本
└── requirements.txt
```

### 训练流程

```
Qwen2.5-1.5B-Instruct
      │
      ▼  Stage 1: Agent SFT（冷启动）
  LoRA 微调（glaive-function-calling-v2，5k 条）
      │  merge LoRA → outputs/sft_merged/
      ▼  Stage 2a: 单轮 Agent GRPO
  grpo_train.py（50 个可验证代码任务）
      │
      ▼  Stage 2b: 多轮 Agent GRPO（扩展）
  multi_turn_grpo.py（支持多轮工具调用 episode）
```

### 算法

**Stage 1 — Agent SFT 冷启动**

- 数据集：`glaiveai/glaive-function-calling-v2`（5k 条函数调用对话）
- 将 glaive 格式（USER/ASSISTANT/FUNCTION RESPONSE）转换为 Qwen ChatML 格式
- LoRA rank=64，训练 2 epoch

**Stage 2a — 单轮 Agent GRPO**

使用 50 个**结果可精确验证**的代码执行任务（涵盖数学计算、字符串处理、算法题、数据处理），训练模型学会调用 `python_exec` 生成并执行代码。

Reward 函数：

| 分量 | 权重 | 说明 |
|------|------|------|
| `tool_call_valid` | +0.3 | 能从输出中解析出合法工具调用 |
| `tool_selection` | +0.2 | 选择了正确的工具（如 `python_exec`）|
| `task_completion` | +1.0 | **实际执行工具后**结果匹配预期答案 |
| `overlong_penalty`| -0.5 | 输出超过 300 词时惩罚 |
| `repetition_pen` | -1.0 | 严重重复惩罚 |

`task_completion` 是关键：Reward 函数真实执行 `SafeToolExecutor`，对比工具输出与预期答案，实现**基于执行结果的奖励**。

**Stage 2b — 多轮 Agent GRPO**

`multi_turn_grpo.py` 扩展了单轮训练，支持完整的多轮 agent episode：

```
prompt
  └─→ LLM 生成（含工具调用）
        └─→ 执行工具，将结果追加到上下文
              └─→ LLM 继续生成（最多 8 轮）
                    └─→ 计算最终 Reward
```

评测时运行完整 episode（`run_agent_episode`），对比工具执行结果与预期答案。

**工具调用解析（tools.py）**

支持三种格式的工具调用解析：
1. JSON `{"name": "xxx", "arguments": {...}}` 格式
2. 完整 JSON 文本
3. `<tool_call>{...}</tool_call>` 标签格式

`SafeToolExecutor` 在临时目录中沙箱化执行所有工具，限制超时（默认 5 秒）和输出长度（500 字符）。

### 使用方法

```bash
cd agent_rl
pip install -r requirements.txt

# Stage 1: SFT 冷启动（工具调用格式学习）
python sft_train.py

# Stage 2a: 单轮 Agent GRPO
python grpo_train.py

# Stage 2b: 多轮 Agent GRPO（扩展训练）
python multi_turn_grpo.py

# 评测
python evaluate.py
```

WandB 关键指标：

- `reward/task_complete`：任务完成率（核心指标）
- `reward/tool_valid`：工具调用格式合规率
- `reward/tool_select`：工具选择准确率
- `eval_mt/accuracy`：多轮评测准确率

---

## 依赖与环境

| 项目 | 核心依赖 |
|------|---------|
| simple_agent | `openai`, `langgraph`, `langchain-openai` |
| simple_posttrain | `torch`, `transformers`, `trl>=0.15`, `peft`, `wandb` |
| agent_rl | `torch`, `transformers`, `trl>=0.15`, `peft`, `wandb` |

推荐环境：CUDA 12.1，Python 3.10，单卡 A100（40GB）可运行所有训练脚本。

## 学习路径

```
simple_agent          →    simple_posttrain      →    agent_rl
（理解 Agent 机制）       （学会后训练流程）          （融合两者：训练会用工具的模型）
```

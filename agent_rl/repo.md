# Agent RL 后训练：项目分析与实现方案

## 一、Open-AgentRL 算法原理与工程实现

### 1.1 项目定位

Open-AgentRL 是基于 VeRL 框架的开源 Agent 强化学习系统，包含两个核心研究方向：
- **RLAnything**: 闭环 RL 优化，动态协调策略、奖励模型和环境
- **DemyAgent**: 聚焦 Agentic 推理，使用真实轨迹和选择性工具使用

**核心成果**：4B 模型通过有效 RL + 工具使用，在数学推理上匹配 14B-32B 模型表现。

### 1.2 GRPO 算法（Group Relative Policy Optimization）

GRPO 的核心思想是**去掉 Critic 模型**，用组内相对奖励替代价值函数估计：

**优势估计**：
```
对每个 prompt，生成 N 个响应（如 N=16），计算各自奖励 r_i
组内均值 μ = mean(r_1, ..., r_N)
组内标准差 σ = std(r_1, ..., r_N)
每个响应的优势 A_i = (r_i - μ) / (σ + ε)
```

**策略梯度损失（带不对称裁剪）**：
```
ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
L = -E_t[min(ρ_t * A_t, clip(ρ_t, 1-ε_low, 1+ε_high) * A_t)]

其中 ε_low=0.2, ε_high=0.28（不对称裁剪，鼓励更多探索）
```

**Token-level 损失聚合（TCR）**：
- `loss_agg_mode="token-mean"`: 梯度信号均匀分布到所有响应 token
- 对比 `sequence-mean`: 短序列和长序列对梯度贡献相同

**优势**：
- 无需训练 Critic 模型，显存减半
- 基于结果的奖励，简单有效
- 动态采样（DAPO）：过滤同质组（全对/全错），聚焦有区分度的学习

### 1.3 多轮 Agent 交互

Open-AgentRL 的多轮训练流程：

```
状态机: PENDING → RUNNING → TOOL_CALLING → RUNNING → ... → COMPLETED

每个 episode:
1. 模型生成响应（可能包含工具调用）
2. 解析工具调用 → 通过 SandboxFusion 执行
3. 工具结果作为新消息追加到上下文
4. 继续生成，直到最大轮次(16)或停止条件
5. 计算最终奖励（结果正确性）
```

**关键实现细节**：
- 使用 vLLM 进行异步 rollout 生成
- SandboxFusion 提供沙箱化代码执行环境
- 对话格式使用 Hermes 模板
- `completion_mask` 只覆盖模型生成的 token，环境注入的 token（工具结果）不参与梯度计算

### 1.4 奖励函数设计

```python
# 多数据集奖励计算
def compute_score(data_source, solution_str, ground_truth, extra_info):
    if 'code' in data_source:
        result = code_math.compute_score(solution_str, ground_truth)  # 执行验证
    else:
        result = math_dapo.compute_score(solution_str, ground_truth)  # LaTeX 匹配

    # 工具使用调整
    num_turns = int(extra_info.get("num_turns", 0))
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1  # 鼓励使用工具
        result["score"] = float(min(-0.6, result["score"] + tool_call_reward))

    return result

# 奖励组件:
# 1. 结果奖励: +1.0（正确）或 -1.0（错误）
# 2. 超长惩罚: -exceed_len / buffer_len * penalty_factor
# 3. 工具使用奖励: 对错误答案，使用了工具的给予较小惩罚
```

### 1.5 训练配置（DemyAgent 4B 示例）

```yaml
模型: Qwen3-4B-RA-SFT
数据: 30K Open-AgentRL
批大小: 64 prompts × 16 generations = 1024 rollouts/step
算法: GRPO + 不对称裁剪 + Token-level 聚合
学习率: 1e-6
GPU: 8×A100, TP=4
多轮深度: 16 turns
最大响应长度: 20480 tokens
```

### 1.6 工程架构

```
Ray TaskRunner
├── Actor/Rollout Worker (FSDP + vLLM)
├── Critic Worker (GRPO 模式下省略)
├── Reward Model Worker
└── Reference Policy Worker (KL 约束)
```

---

## 二、OpenClaw-RL 持续学习系统分析

### 2.1 核心创新

OpenClaw-RL 将**日常使用对话自动转化为训练数据**，实现 Agent 的持续优化：
- 用户/环境的反馈作为隐式奖励信号
- 对话不中断，训练在后台异步进行
- 模型越用越好用

### 2.2 四组件异步架构

```
┌─────────────────────────────────────────────────────────┐
│  组件1: Agent 服务 (FastAPI Proxy)                       │
│  - OpenAI 兼容 API (port 30000)                         │
│  - 转发请求到 SGLang 推理服务器                           │
│  - 提取 per-token log-probabilities                      │
├─────────────────────────────────────────────────────────┤
│  组件2: Rollout 收集                                      │
│  - 分类 turn: "main"(可训练) vs "side"(不训练)            │
│  - 缓冲响应数据 + token 化结果                            │
│  - 等待 next-state 反馈                                  │
├─────────────────────────────────────────────────────────┤
│  组件3: PRM 奖励评估                                      │
│  - 用 next-state 作为证据评判上一轮响应                    │
│  - m=3 次独立评估 + 多数投票                              │
│  - 返回 +1(好) / -1(差) / 0(中性)                        │
├─────────────────────────────────────────────────────────┤
│  组件4: 策略训练 (Slime RL Trainer)                       │
│  - 后台持续训练                                           │
│  - 权重更新时暂停数据收集                                  │
│  - PPO/GRPO 损失 + KL 正则化                              │
└─────────────────────────────────────────────────────────┘
```

**关键特性**：四个组件完全异步，互不阻塞。

### 2.3 反馈循环

```
用户请求 → API 代理响应 + 收集 log-probs
       → 用户/环境发送下一条消息(next-state)
       → PRM 用 next-state 评判上一轮响应
       → 多数投票 → 标量奖励 (+1/-1/0)
       → 训练样本入队 → GRPO 训练
       → 权重更新 → 更好的响应
       → 循环继续...
```

**隐式奖励信号来源**：
- 用户点赞/踩 → PRM 判为 +1/-1
- 环境成功/失败 → PRM 解读为 +1/-1
- 用户纠正/修改请求 → PRM 检测为 -1
- 工具执行成功 → PRM 评分 +1

### 2.4 两种学习范式

**方式 A: Binary RL (GRPO)**
```
奖励: 标量 (+1, -1, 0)
优势: A_t = r（均匀广播到所有响应 token）
损失: PPO 裁剪替代损失 + KL 正则化 (β=0.02)
裁剪: ε_low=0.2, ε_high=0.28（不对称）
```

**方式 B: On-Policy Distillation (OPD)**
```
从 next-state 提取 hindsight hints（事后提示）
构造增强 prompt: 原始 prompt + hint
教师模型在增强 prompt 上生成 log-probs
Token-level 优势: A_t = log π_teacher(a_t|s+hint) - log π_θ(a_t|s)
更丰富的梯度信号，逐 token 的方向性监督
```

### 2.5 持续学习的挑战与解决方案

| 挑战 | 问题 | 解决方案 |
|------|------|----------|
| 延迟反馈 | 奖励需等待 next-state 到达 | 会话结束时强制提交最后一轮 |
| 信号模糊 | 并非所有 next-state 明确指示成败 | PRM 多数投票 (m=3) 降噪 |
| 奖励稀疏 | 二元 +1/-1 信号稀疏 | OPD: 提取 hindsight hints 提供 token-level 信号 |
| 训练-服务同步 | 权重更新可能导致数据损坏 | 优雅暂停/恢复机制，清除待处理记录 |
| 会话感知 | 多轮需要正确排序和跟踪 | Session ID + turn number + 待处理数据结构 |
| 灾难性遗忘 | 持续更新可能降低基础能力 | KL 损失防止偏离参考模型 (β=0.02) |

### 2.6 数据管理

```python
# 会话跟踪
self._turn_counts[session_id]           # 轮次计数
self._pending_records[session_id]       # 缓冲记录
self._pending_turn_data[session_id]     # 持有轮次数据
self._prm_tasks[session_id]            # PRM 评估任务
self._session_effective[session_id]     # 有效样本计数

# 队列缓冲
output_queue = Queue(maxsize=100000)
# 收集 16 个样本后开始一轮训练 (rollout_batch_size=16)

# 优雅暂停
pause_submission()   # 暂停收集，清除待处理
# ... 权重更新 ...
resume_submission()  # 恢复收集
```

---

## 三、硬件约束分析与模型选择

### 3.1 硬件限制

| 资源 | 可用 | Open-AgentRL 需求 | 差距 |
|------|------|-------------------|------|
| GPU 数量 | 1 | 8 | 8x |
| GPU 显存 | 24GB(3090)/80GB(A100) | 80GB×8 | 显著不足 |
| 张量并行 | 不可用 | TP=4 | 无法使用 |
| 模型规模 | ≤3B | 4B-7B | 需缩小 |

### 3.2 模型选择：Qwen2.5-1.5B-Instruct

**选择理由**：
1. **Instruct 版本**：已有工具调用能力，SFT 只需强化格式一致性
2. **1.5B 参数量**：bf16 约 3GB 显存，为 RL 训练留出充足空间
3. **GRPO 显存预算（3090 24GB）**：
   - 模型权重: 3GB
   - LoRA + 优化器: 0.5GB
   - KV-cache (4 seq × 2048 tokens): 4GB
   - 梯度 + 激活 (gradient checkpointing): 4GB
   - 总计: ~12GB，余量 12GB
4. **对比 3B**：GRPO 生成阶段占 ~20GB，余量不足

### 3.3 关键缩减策略

| 原始配置 | 缩减配置 | 理由 |
|----------|----------|------|
| num_generations=16 | num_generations=4 | 显存限制 |
| batch_size=64 | batch_size=4 | 单卡限制 |
| max_response_length=20480 | max_completion_length=512 | 显存限制 |
| full fine-tuning | LoRA (r=32) | 显存和稳定性 |
| TP=4, 8×A100 | 单卡 3090/A100 | 硬件限制 |
| SandboxFusion | 本地 subprocess | 简化部署 |

---

## 四、完整实现方案

### 4.1 三阶段渐进式训练

```
Stage 1: Agent SFT 冷启动
  └── glaive-function-calling-v2 (5K 样本)
  └── 强化工具调用格式
  └── LoRA r=64, 2 epochs

Stage 2: Agent GRPO 强化学习
  ├── Phase 1: 单轮工具调用 GRPO
  │   └── 标准 TRL GRPOTrainer
  │   └── Reward: 解析+执行工具调用，验证结果
  └── Phase 2: 多轮 Agent GRPO
      └── 子类化 GRPOTrainer
      └── 多轮交互循环 + completion_mask

Stage 3: 持续学习（未来）
  └── vLLM 部署 → FastAPI 代理 → JSONL 缓冲 → 定期微调
```

### 4.2 Reward 函数设计

**单轮模式（Phase 1）**：
```python
def agent_reward_fn(completions, **kwargs):
    for completion in completions:
        r = 0.0
        # 1. 工具调用合法性：能解析出有效工具调用 → +0.3
        tool_calls = parse_tool_calls(completion)
        if tool_calls: r += 0.3

        # 2. 工具选择正确性：选对了工具 → +0.2
        if correct_tool_selected: r += 0.2

        # 3. 任务完成度：执行工具后结果匹配预期 → +1.0
        if execute_and_verify(tool_calls, expected): r += 1.0

        # 4. 超长惩罚 → -0.5
        if too_long: r -= 0.5
```

**多轮模式（Phase 2）**：
```python
def multi_turn_reward_fn(trajectory, task):
    r = 0.0
    if task_completed(trajectory.final_state, task.expected): r += 1.0
    r += -0.1 * trajectory.num_turns  # 效率惩罚
    return r
```

### 4.3 多轮 GRPO 关键设计

```python
class MultiTurnGRPOTrainer(GRPOTrainer):
    def _prepare_inputs(self, inputs):
        # 1. 对每个 prompt，运行 agent episode:
        #    generate → parse_tool_calls → execute → append_result → repeat
        # 2. 构建 completion_mask:
        #    模型生成的 token → mask=1（参与梯度计算）
        #    环境注入的 token → mask=0（不参与梯度计算）
        # 3. 计算 ref log-probs, rewards, advantages
        # 4. 返回标准格式供 compute_loss 使用
```

### 4.4 数据来源

**SFT 数据**：`glaiveai/glaive-function-calling-v2`
- 113K 工具调用对话，取 5K 做冷启动
- 包含系统提示 + 用户查询 + 工具调用 + 工具结果 + 最终回答

**RL 任务**：代码执行任务为主
- 基于 HumanEval 子集或自定义编程任务
- 每个任务有可验证的预期输出
- 使用 `python_exec` 工具执行代码并验证

### 4.5 持续学习架构（Stage 3 设计）

简化 OpenClaw-RL 为单卡版本：

```
┌─ vLLM 推理服务 ──────────────────────┐
│  部署 GRPO 训练后的模型               │
│  提供 OpenAI 兼容 API                │
└──────────┬───────────────────────────┘
           │
┌──────────▼───────────────────────────┐
│  FastAPI 代理（轻量级）               │
│  - 拦截对话，提取 log-probs          │
│  - 工具执行结果作为隐式奖励          │
│  - 对话日志写入 JSONL 缓冲           │
└──────────┬───────────────────────────┘
           │ 每 100 条对话触发
┌──────────▼───────────────────────────┐
│  离线 GRPO 微调                       │
│  - 从 JSONL 加载高/低奖励对话         │
│  - LoRA 微调（保持轻量）              │
│  - KL 正则化 (β=0.02) 防止遗忘       │
│  - 更新 vLLM 模型权重                │
└──────────────────────────────────────┘
```

### 4.6 文件结构与实现顺序

```
agent_rl/
├── repo.md                # 本文档
├── config.py              # 集中配置
├── tools.py               # 工具系统（复用 simple_agent）
├── data.py                # 数据加载与格式化
├── rewards.py             # Agent Reward 函数
├── sft_train.py           # Stage 1: Agent SFT
├── grpo_train.py          # Stage 2.1: 单轮 GRPO
├── multi_turn_grpo.py     # Stage 2.2: 多轮 GRPO
├── evaluate.py            # Agent 能力评测
└── requirements.txt       # 依赖
```

实现顺序：config → tools → data → sft_train → rewards → grpo_train → evaluate → multi_turn_grpo

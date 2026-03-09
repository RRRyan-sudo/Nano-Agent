"""
Agent RL 后训练 —— 集中配置
SFT 冷启动 + GRPO 强化学习，提升模型工具调用和任务完成能力
"""
import os

# ─── 模型 ─────────────────────────────────────────────
# Instruct 版本：已具备工具调用能力，SFT 只需强化格式一致性
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# ─── 路径 ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SFT_OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "sft")
GRPO_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "grpo")
SFT_MERGED_DIR  = os.path.join(BASE_DIR, "outputs", "sft_merged")

# ─── WandB ────────────────────────────────────────────
WANDB_PROJECT   = "agent-rl-posttrain"
WANDB_ENTITY    = None

# ─── Agent SFT 超参 ──────────────────────────────────
AGENT_SFT_CONFIG = dict(
    # 数据
    dataset_name          = "glaiveai/glaive-function-calling-v2",
    max_train_samples     = 5000,       # 5K 样本做冷启动
    max_seq_length        = 2048,
    packing               = False,      # 多轮对话不打包

    # 训练
    num_train_epochs      = 2,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,    # 有效 batch = 16
    learning_rate         = 2e-4,
    lr_scheduler_type     = "cosine",
    warmup_ratio          = 0.1,

    # 精度
    bf16                  = True,
    optim                 = "adamw_torch",
    gradient_checkpointing= True,

    # 日志 & 保存
    logging_steps         = 10,
    save_strategy         = "steps",
    save_steps            = 300,
    save_total_limit      = 2,

    # LoRA
    lora_r                = 64,
    lora_alpha            = 128,
    lora_dropout          = 0.05,
    lora_target_modules   = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ─── Agent GRPO 超参 ─────────────────────────────────
AGENT_GRPO_CONFIG = dict(
    # GRPO 核心
    num_generations       = 4,          # 每 prompt 生成 4 个响应做组内比较
    max_completion_length = 512,        # 工具调用 + 推理需要更多 token
    temperature           = 0.7,        # 适当探索
    beta                  = 0.05,       # 较低 KL penalty，Agent 需要更多探索

    # 训练
    num_train_epochs      = 1,
    per_device_train_batch_size = 4,    # 必须 >= num_generations
    gradient_accumulation_steps = 4,    # 有效 batch = 16
    learning_rate         = 3e-6,       # RL 阶段学习率要小
    lr_scheduler_type     = "cosine",
    warmup_ratio          = 0.05,

    # 稳定性
    bf16                  = True,
    gradient_checkpointing= True,
    max_grad_norm         = 0.5,        # 梯度裁剪

    # 日志 & 保存
    logging_steps         = 5,
    save_strategy         = "steps",
    save_steps            = 200,
    save_total_limit      = 2,
    eval_steps            = 50,

    # LoRA（RL 阶段用更小的 rank 更稳定）
    lora_r                = 32,
    lora_alpha            = 64,
    lora_dropout          = 0.05,
    lora_target_modules   = [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
)

# ─── Reward 权重 ──────────────────────────────────────
AGENT_REWARD_WEIGHTS = dict(
    task_completion  = 1.0,    # 任务完成（执行结果匹配预期）
    tool_call_valid  = 0.3,    # 工具调用格式合法
    tool_selection   = 0.2,    # 选择了正确的工具
    overlong_penalty = -0.5,   # 超长惩罚
    repetition_pen   = -1.0,   # 严重重复惩罚
)

# ─── Agent 系统提示词 ─────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to the following tools:

1. **python_exec** - Execute Python code and return the output
2. **shell_exec** - Execute a shell command and return its output
3. **file_read** - Read the contents of a file
4. **file_write** - Write content to a file

When you need to solve a task, think step by step and use the appropriate tool.
Call tools using the function calling format. After receiving tool results, analyze them and continue or provide your final answer.
Always verify your results before giving a final answer."""

# ─── 评估配置 ─────────────────────────────────────────
EVAL_SAMPLE_SIZE = 50    # 评测任务数量

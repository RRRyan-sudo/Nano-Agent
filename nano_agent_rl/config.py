"""
共享配置 —— SFT 和 GRPO 训练脚本均从此导入
修改这里即可统一调整超参和路径
"""
import os

# ─── 模型 ─────────────────────────────────────────────
# 用 Base 模型（非 Instruct）：SFT/RL 提升效果最明显，故事最完整
# 如改回 Instruct，删掉 sft_train.py 里的 setup_chat_template 调用即可
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# ─── 路径 ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SFT_OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "sft")
GRPO_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "grpo")
# SFT 训练后 merge LoRA 得到的完整模型，作为 GRPO 的起点
SFT_MERGED_DIR  = os.path.join(BASE_DIR, "outputs", "sft_merged")

# ─── WandB ────────────────────────────────────────────
WANDB_PROJECT   = "qwen-posttraining"
WANDB_ENTITY    = None   # 填写你的 wandb 用户名，None 则用默认

# ─── SFT 超参 ─────────────────────────────────────────
SFT_CONFIG = dict(
    # 数据
    # Base 模型从零开始学对话格式，需要更多数据，用 20k 条
    dataset_name          = "HuggingFaceH4/ultrachat_200k",
    dataset_split         = "train_sft[:20000]",   # 2 万条，约 2-3h

    # 序列
    max_seq_length        = 2048,
    packing               = True,   # 短序列打包提升利用率

    # 训练
    num_train_epochs      = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,   # 有效 batch = 16
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
    save_steps            = 400,
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

# ─── GRPO 超参 ────────────────────────────────────────
GRPO_CONFIG = dict(
    # 数据
    dataset_name          = "openai/gsm8k",
    dataset_config        = "main",
    dataset_split         = "train",
    max_train_samples     = 2400,   # 2400/4=600步，对1.5B模型足够收敛

    # GRPO 核心
    num_generations       = 4,      # 每 prompt 并行生成 4 个回答做组内比较
    max_completion_length = 256,    # GSM8K reasoning+answer 通常 120-200 token 足够
    temperature           = 0.8,    # 适当探索
    beta                  = 0.1,    # KL penalty 系数

    # 训练
    num_train_epochs      = 1,
    per_device_train_batch_size = 4,    # 必须 >= num_generations=4 且可整除
    gradient_accumulation_steps = 4,   # 有效 batch = 4×4 = 16，保持不变
    learning_rate         = 5e-6,   # RL 阶段学习率要小
    lr_scheduler_type     = "cosine",
    warmup_ratio          = 0.05,

    # 精度 & 稳定性
    bf16                  = True,
    gradient_checkpointing= True,
    max_grad_norm         = 0.5,    # 梯度裁剪，RL 训练更重要

    # 日志 & 保存
    logging_steps         = 5,
    save_strategy         = "steps",
    save_steps            = 200,
    save_total_limit      = 2,
    eval_steps            = 50,     # 每 50 step 在固定题目上评测

    # LoRA（RL 阶段用更小的 rank 更稳定）
    lora_r                = 32,
    lora_alpha            = 64,
    lora_dropout          = 0.05,
    lora_target_modules   = [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
)

# ─── Reward 权重 ──────────────────────────────────────
REWARD_WEIGHTS = dict(
    answer_correct   = 1.0,   # 答案数字完全匹配
    format_complete  = 0.5,   # 同时包含 <reasoning> 和 <answer> 标签
    step_quality     = 0.3,   # 推理步骤字数适当 (50-400 字)
    repetition_pen   = -1.0,  # 严重重复惩罚
)

# ─── GSM8K 推理格式 Prompt ───────────────────────────
MATH_SYSTEM_PROMPT = """You are a math reasoning expert. Solve problems step by step.

Always use this exact format:
<reasoning>
Write your step-by-step solution here.
</reasoning>
<answer>
[final numeric answer only, no units]
</answer>"""

# ─── 评估用固定题目（GRPO 训练中每 eval_steps 跑一次） ──
# 从 GSM8K test 集中挑选有代表性的 20 道题
EVAL_SAMPLE_SIZE = 20

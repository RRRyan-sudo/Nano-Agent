"""
Stage 2: GRPO（Group Relative Policy Optimization）
模型起点: SFT merged 模型 (outputs/sft_merged)
数据: openai/gsm8k（小学数学应用题）
Reward: 答案正确 + 推理格式 + 步骤质量 - 重复惩罚
目标: 通过 RL 强化数学推理能力，验证 SFT→RL 的提升

运行方式:
    python grpo_train.py

WandB 项目: qwen-posttraining / run: qwen1.5b-grpo

关键指标（WandB）：
    reward/mean          应稳步上升
    reward/answer_score  答案准确率，最直观的改善信号
    reward/format_score  格式合规率
    train/kl             KL 散度，不应持续增大
    train/completion_len 生成长度，不应爆炸增长
    eval/accuracy        定期评测准确率
"""

import os
import re
import time
import json
import wandb
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback

from config import (
    MODEL_NAME, SFT_MERGED_DIR, GRPO_OUTPUT_DIR,
    WANDB_PROJECT, WANDB_ENTITY, GRPO_CONFIG, REWARD_WEIGHTS,
    MATH_SYSTEM_PROMPT, EVAL_SAMPLE_SIZE,
)

os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Reward 函数组件
# ──────────────────────────────────────────────────────────────

def normalize_number(s: str) -> str:
    """统一数字格式：去除逗号、空格，统一小数点。"""
    return s.replace(",", "").replace(" ", "").strip()


def extract_answer(text: str) -> str | None:
    """
    从模型输出中提取最终答案数字。
    优先级: <answer>标签 > '#### 数字' GSM8K格式 > 最后出现的数字
    """
    # 1. <answer> 标签
    m = re.search(r"<answer>\s*([\d,\.]+)", text)
    if m:
        return normalize_number(m.group(1))

    # 2. GSM8K 风格：#### 42
    m = re.search(r"####\s*([\d,\.]+)", text)
    if m:
        return normalize_number(m.group(1))

    # 3. fallback：文本中最后一个数字
    nums = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return normalize_number(nums[-1])

    return None


def extract_gt_answer(answer_str: str) -> str | None:
    """从 GSM8K 数据集的 answer 字段提取 #### 后的数字。"""
    m = re.search(r"####\s*([\d,\.]+)", answer_str)
    if m:
        return normalize_number(m.group(1))
    return None


def answer_reward(completion: str, gt_answer: str) -> float:
    """答案正确性奖励：完全匹配 +1.0，否则 0。"""
    pred = extract_answer(completion)
    if pred is None or gt_answer is None:
        return 0.0
    return REWARD_WEIGHTS["answer_correct"] if pred == gt_answer else 0.0


def format_reward(completion: str) -> float:
    """格式合规奖励：同时包含 <reasoning> 和 <answer> 标签。"""
    has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", completion, re.DOTALL))
    has_answer    = bool(re.search(r"<answer>.*?</answer>",    completion, re.DOTALL))
    if has_reasoning and has_answer:
        return REWARD_WEIGHTS["format_complete"]
    # 部分奖励：只有其中一个
    if has_reasoning or has_answer:
        return REWARD_WEIGHTS["format_complete"] * 0.3
    return 0.0


def step_quality_reward(completion: str) -> float:
    """
    推理步骤质量奖励：
    - <reasoning> 内容长度适中（50-400字）说明有实质推理
    - 过短 → 没有推理；过长 → 可能在凑字数
    """
    m = re.search(r"<reasoning>(.*?)</reasoning>", completion, re.DOTALL)
    if m:
        length = len(m.group(1).strip())
        if 50 <= length <= 400:
            return REWARD_WEIGHTS["step_quality"]
        elif length > 400:
            return REWARD_WEIGHTS["step_quality"] * 0.5   # 太长打折
    return 0.0


def repetition_penalty(completion: str) -> float:
    """
    重复惩罚：词级别 unique ratio < 0.4 则触发。
    类比 embodied RL 中的 action smoothness penalty。
    """
    words = completion.split()
    if len(words) < 15:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.4:
        return REWARD_WEIGHTS["repetition_pen"]
    return 0.0


# ──────────────────────────────────────────────────────────────
# 组合 Reward 函数（传给 GRPOTrainer）
# ──────────────────────────────────────────────────────────────

# 全局存储每 step 各分量均值，供 WandB Callback 读取
_reward_components_buffer = {}


def math_reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """
    GRPO reward 函数签名：接收一批 completions，返回对应 reward 列表。
    kwargs 中包含 dataset 的其他字段（我们传入 gt_answer）。
    """
    gt_answers = kwargs.get("gt_answer", [None] * len(completions))

    rewards     = []
    ans_scores  = []
    fmt_scores  = []
    step_scores = []
    rep_scores  = []

    for completion, gt in zip(completions, gt_answers):
        a = answer_reward(completion, gt)
        f = format_reward(completion)
        s = step_quality_reward(completion)
        r = repetition_penalty(completion)
        total = a + f + s + r

        rewards.append(total)
        ans_scores.append(a)
        fmt_scores.append(f)
        step_scores.append(s)
        rep_scores.append(r)

    # 写入缓冲，WandB Callback 会读取并 log
    _reward_components_buffer.update({
        "reward/answer_score": float(np.mean(ans_scores)),
        "reward/format_score": float(np.mean(fmt_scores)),
        "reward/step_score":   float(np.mean(step_scores)),
        "reward/rep_penalty":  float(np.mean(rep_scores)),
        "reward/mean":         float(np.mean(rewards)),
        "reward/std":          float(np.std(rewards)),
        # 准确率：answer_score / max_answer_reward
        "reward/accuracy":     float(np.mean([a > 0 for a in ans_scores])),
    })

    return rewards


# ──────────────────────────────────────────────────────────────
# 定期评测（在固定题目上算准确率）
# ──────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, eval_dataset, device, step: int):
    """
    在 eval_dataset（固定 20 道题）上做 greedy 推理，计算准确率。
    结果 log 到 wandb。
    """
    model.eval()
    correct = 0
    rows = []   # 用于 wandb.Table

    with torch.no_grad():
        for sample in eval_dataset:
            messages = [
                {"role": "system",  "content": MATH_SYSTEM_PROMPT},
                {"role": "user",    "content": sample["question"]},
            ]
            text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,          # greedy decode for eval
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )
            gt  = extract_gt_answer(sample["answer"])
            pred = extract_answer(response)
            is_correct = (pred == gt) if (pred and gt) else False
            if is_correct:
                correct += 1
            rows.append([sample["question"][:80], gt, pred or "N/A", "✓" if is_correct else "✗"])

    accuracy = correct / len(eval_dataset)

    # wandb Table（可在 UI 里直接看题目 / 答案 / 对错）
    table = wandb.Table(
        columns=["question", "gt_answer", "pred_answer", "correct"],
        data=rows,
    )
    wandb.log({"eval/accuracy": accuracy, "eval/predictions": table}, step=step)
    print(f"[Eval @ step {step}] accuracy = {accuracy:.3f} ({correct}/{len(eval_dataset)})")

    model.train()
    return accuracy


# ──────────────────────────────────────────────────────────────
# WandB Callback
# ──────────────────────────────────────────────────────────────

class GRPOWandBCallback(TrainerCallback):
    """
    注入额外 WandB 指标：
    - 各 reward 分量均值
    - completion 平均长度
    - 每 eval_steps 运行一次准确率评测
    """

    def __init__(self, eval_dataset, tokenizer, eval_steps: int):
        self.eval_dataset = eval_dataset
        self.tokenizer    = tokenizer
        self.eval_steps   = eval_steps
        self._comp_lengths = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return

        extra = {}

        # reward 各分量
        if _reward_components_buffer:
            extra.update(_reward_components_buffer)

        # completion 平均长度（从 _comp_lengths 缓冲读取）
        if self._comp_lengths:
            extra["train/completion_length"] = float(np.mean(self._comp_lengths))
            self._comp_lengths.clear()

        # KL 散度（TRL 内部可能已经 log，这里做一个兜底）
        if "kl" in logs:
            extra["train/kl"] = logs["kl"]

        if extra:
            wandb.log(extra, step=state.global_step)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # 定期评测
        if (
            state.is_world_process_zero
            and state.global_step > 0
            and state.global_step % self.eval_steps == 0
            and model is not None
        ):
            device = next(model.parameters()).device
            run_eval(model, self.tokenizer, self.eval_dataset, device, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        wandb.run.summary["train/total_steps"] = state.global_step
        # 最终评测
        if model is not None:
            device = next(model.parameters()).device
            final_acc = run_eval(
                model, self.tokenizer, self.eval_dataset, device, state.global_step
            )
            wandb.run.summary["eval/final_accuracy"] = final_acc


# ──────────────────────────────────────────────────────────────
# 数据处理
# ──────────────────────────────────────────────────────────────

def load_gsm8k(tokenizer, max_samples: int):
    """
    加载 GSM8K，将每条样本转换为：
    - prompt: 用 chat template 格式化的对话文本（只有 system + user）
    - gt_answer: ground truth 数字字符串
    """
    print(f"Loading GSM8K train (max {max_samples} samples)...")
    dataset = load_dataset(
        GRPO_CONFIG["dataset_name"],
        GRPO_CONFIG["dataset_config"],
        split=GRPO_CONFIG["dataset_split"],
    )
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    def process(example):
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user",   "content": example["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        gt = extract_gt_answer(example["answer"])
        return {"prompt": prompt, "gt_answer": gt or ""}

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x["gt_answer"] != "")
    print(f"GSM8K train size: {len(dataset)}")
    return dataset


def load_gsm8k_eval(tokenizer):
    """加载 GSM8K test 集的前 EVAL_SAMPLE_SIZE 条用于定期评测。"""
    dataset = load_dataset(
        GRPO_CONFIG["dataset_name"],
        GRPO_CONFIG["dataset_config"],
        split="test",
    )
    dataset = dataset.select(range(EVAL_SAMPLE_SIZE))
    return dataset   # 保留原始字段，run_eval 直接用 question / answer


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    # ── 0. WandB ─────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="qwen1.5b-grpo",
        config={
            "model":   SFT_MERGED_DIR,
            "stage":   "GRPO",
            "dataset": "gsm8k",
            **GRPO_CONFIG,
            **{f"reward/{k}": v for k, v in REWARD_WEIGHTS.items()},
        },
    )

    # ── 1. Tokenizer ─────────────────────────────────────────
    model_path = SFT_MERGED_DIR if os.path.exists(SFT_MERGED_DIR) else MODEL_NAME
    if not os.path.exists(SFT_MERGED_DIR):
        print(f"WARNING: SFT merged model not found at {SFT_MERGED_DIR}")
        print(f"         Falling back to base model: {MODEL_NAME}")

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "left"   # GRPO 生成时左 padding

    # ── 2. 数据集 ────────────────────────────────────────────
    train_dataset = load_gsm8k(tokenizer, GRPO_CONFIG["max_train_samples"])
    eval_dataset  = load_gsm8k_eval(tokenizer)

    # ── 3. 模型 ──────────────────────────────────────────────
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── 4. LoRA（RL 阶段用更小的 rank，更稳定） ───────────────
    peft_config = LoraConfig(
        r=GRPO_CONFIG["lora_r"],
        lora_alpha=GRPO_CONFIG["lora_alpha"],
        target_modules=GRPO_CONFIG["lora_target_modules"],
        lora_dropout=GRPO_CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 5. GRPO 配置 ──────────────────────────────────────────
    grpo_args = GRPOConfig(
        output_dir=GRPO_OUTPUT_DIR,

        # GRPO 核心
        num_generations=GRPO_CONFIG["num_generations"],
        max_completion_length=GRPO_CONFIG["max_completion_length"],
        temperature=GRPO_CONFIG["temperature"],
        beta=GRPO_CONFIG["beta"],

        # 训练
        num_train_epochs=GRPO_CONFIG["num_train_epochs"],
        per_device_train_batch_size=GRPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=GRPO_CONFIG["gradient_accumulation_steps"],
        learning_rate=GRPO_CONFIG["learning_rate"],
        lr_scheduler_type=GRPO_CONFIG["lr_scheduler_type"],
        warmup_ratio=GRPO_CONFIG["warmup_ratio"],

        # 稳定性
        bf16=GRPO_CONFIG["bf16"],
        gradient_checkpointing=GRPO_CONFIG["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=GRPO_CONFIG["max_grad_norm"],

        # 日志 & 保存
        logging_steps=GRPO_CONFIG["logging_steps"],
        save_strategy=GRPO_CONFIG["save_strategy"],
        save_steps=GRPO_CONFIG["save_steps"],
        save_total_limit=GRPO_CONFIG["save_total_limit"],

        # WandB
        report_to="wandb",
        run_name="qwen1.5b-grpo",
    )

    # ── 6. Callback ───────────────────────────────────────────
    grpo_callback = GRPOWandBCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        eval_steps=GRPO_CONFIG["eval_steps"],
    )

    # ── 7. Trainer ───────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=math_reward_fn,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[grpo_callback],
    )

    # ── 8. 训练 ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Starting GRPO Training")
    print(f"  Base   : {model_path}")
    print(f"  Data   : GSM8K ({len(train_dataset)} prompts)")
    print(f"  Gens   : {GRPO_CONFIG['num_generations']} per prompt")
    print(f"  Reward : answer({REWARD_WEIGHTS['answer_correct']}) + format({REWARD_WEIGHTS['format_complete']}) + step({REWARD_WEIGHTS['step_quality']}) + rep({REWARD_WEIGHTS['repetition_pen']})")
    print(f"  β (KL) : {GRPO_CONFIG['beta']}")
    print("="*60 + "\n")

    trainer.train()

    # ── 9. 保存最终模型 ───────────────────────────────────────
    final_path = os.path.join(GRPO_OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"GRPO model saved → {final_path}")

    wandb.run.summary["grpo_output_path"] = final_path
    wandb.finish()
    print("\nGRPO training complete!")


if __name__ == "__main__":
    main()

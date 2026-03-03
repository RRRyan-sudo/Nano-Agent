"""
Stage 1: SFT（监督微调）
模型: Qwen2.5-1.5B-Instruct
数据: HuggingFaceH4/ultrachat_200k（通用多轮对话）
方法: LoRA 微调 + packing
目标: 让模型更好地跟随指令，为 GRPO 阶段提供稳定的起点

运行方式:
    python sft_train.py

训练后自动 merge LoRA → outputs/sft_merged/（GRPO 的起点）
WandB 项目: qwen-posttraining / run: qwen1.5b-sft
"""

import os
import time
import math
import wandb
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from config import (
    MODEL_NAME, SFT_OUTPUT_DIR, SFT_MERGED_DIR,
    WANDB_PROJECT, WANDB_ENTITY, SFT_CONFIG,
)

os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)
os.makedirs(SFT_MERGED_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# WandB Callback：记录额外指标（tokens/sec、perplexity）
# ──────────────────────────────────────────────────────────────
class SFTWandBCallback(TrainerCallback):
    """在标准 HF Trainer 日志基础上，补充 tokens/sec 和 perplexity。"""

    def __init__(self):
        self._step_start_time = None
        self._tokens_this_step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return

        extra = {}

        # Perplexity（从 loss 推算）
        if "loss" in logs:
            try:
                extra["train/perplexity"] = math.exp(logs["loss"])
            except OverflowError:
                extra["train/perplexity"] = float("inf")

        # Tokens per second（粗略估算：batch_size * seq_len / elapsed）
        if self._step_start_time is not None:
            elapsed = time.time() - self._step_start_time
            if elapsed > 0:
                # 有效 token 数 = batch * grad_accum * seq_len（packing 下约等于 max_seq_length）
                tokens = (
                    args.per_device_train_batch_size
                    * args.gradient_accumulation_steps
                    * args.max_seq_length  # SFTConfig 上有这个字段
                )
                extra["train/tokens_per_second"] = tokens / elapsed

        if extra:
            wandb.log(extra, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        # 记录最终 loss 对应的 perplexity 到 summary
        if state.log_history:
            last_loss = next(
                (h["loss"] for h in reversed(state.log_history) if "loss" in h), None
            )
            if last_loss is not None:
                try:
                    wandb.run.summary["train/final_perplexity"] = math.exp(last_loss)
                except OverflowError:
                    pass
        wandb.run.summary["train/total_steps"] = state.global_step


# ──────────────────────────────────────────────────────────────
# Chat Template（Base 模型需要手动设置）
# ──────────────────────────────────────────────────────────────
# Qwen2 系列统一使用 ChatML 格式
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{% endif %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def setup_chat_template(tokenizer):
    """
    为 Base 模型补充 ChatML 模板。
    Instruct 模型的 tokenizer 已内置模板，此函数对其无害（不会覆盖已有模板）。
    """
    if tokenizer.chat_template is None:
        print("chat_template not found in tokenizer → setting Qwen ChatML template")
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    else:
        print("chat_template already exists in tokenizer, skipping setup")


# ──────────────────────────────────────────────────────────────
# 数据处理
# ──────────────────────────────────────────────────────────────
def load_sft_dataset(tokenizer):
    """加载 ultrachat_200k 并转换为 chat format 文本。"""
    print(f"Loading dataset: {SFT_CONFIG['dataset_name']} [{SFT_CONFIG['dataset_split']}]")
    dataset = load_dataset(
        SFT_CONFIG["dataset_name"],
        split=SFT_CONFIG["dataset_split"],
    )

    def format_chat(example):
        """将 messages 列表用 chat template 转为单个字符串。"""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
    # 过滤掉过短或过长的样本
    dataset = dataset.filter(
        lambda x: 50 < len(x["text"]) < SFT_CONFIG["max_seq_length"] * 4
    )
    print(f"Dataset size after filtering: {len(dataset)}")
    return dataset


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main():
    # ── 0. WandB 初始化 ──────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="qwen1.5b-sft",
        config={
            "model": MODEL_NAME,
            "stage": "SFT",
            **SFT_CONFIG,
        },
    )

    # ── 1. Tokenizer ─────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    setup_chat_template(tokenizer)  # Base 模型需要此步骤

    # ── 2. 数据集 ────────────────────────────────────────────
    dataset = load_sft_dataset(tokenizer)

    # 划分 train / eval（保留 200 条做验证曲线）
    split = dataset.train_test_split(test_size=200, seed=42)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    print(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

    # ── 3. 模型 ──────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False   # gradient checkpointing 时关闭 KV cache

    # ── 4. LoRA ──────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=SFT_CONFIG["lora_r"],
        lora_alpha=SFT_CONFIG["lora_alpha"],
        target_modules=SFT_CONFIG["lora_target_modules"],
        lora_dropout=SFT_CONFIG["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 预期: trainable params ~40M / all ~1.5B (~2.7%)

    # ── 5. 训练配置 ───────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,

        # 训练
        num_train_epochs=SFT_CONFIG["num_train_epochs"],
        per_device_train_batch_size=SFT_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=SFT_CONFIG["gradient_accumulation_steps"],
        learning_rate=SFT_CONFIG["learning_rate"],
        lr_scheduler_type=SFT_CONFIG["lr_scheduler_type"],
        warmup_ratio=SFT_CONFIG["warmup_ratio"],

        # 精度
        bf16=SFT_CONFIG["bf16"],
        optim=SFT_CONFIG["optim"],
        gradient_checkpointing=SFT_CONFIG["gradient_checkpointing"],

        # 序列 & packing
        max_seq_length=SFT_CONFIG["max_seq_length"],
        packing=SFT_CONFIG["packing"],
        dataset_text_field="text",

        # 评估
        eval_strategy="steps",
        eval_steps=200,
        per_device_eval_batch_size=2,

        # 日志 & 保存
        logging_steps=SFT_CONFIG["logging_steps"],
        save_strategy=SFT_CONFIG["save_strategy"],
        save_steps=SFT_CONFIG["save_steps"],
        save_total_limit=SFT_CONFIG["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # WandB
        report_to="wandb",
        run_name="qwen1.5b-sft",
    )

    # ── 6. Trainer ───────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[SFTWandBCallback()],
    )

    # ── 7. 训练 ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Starting SFT Training")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Data  : {SFT_CONFIG['dataset_name']} ({len(train_dataset)} samples)")
    print(f"  Epochs: {SFT_CONFIG['num_train_epochs']}")
    print(f"  Effective batch: {SFT_CONFIG['per_device_train_batch_size'] * SFT_CONFIG['gradient_accumulation_steps']}")
    print("="*60 + "\n")

    trainer.train()

    # ── 8. 保存 LoRA adapter ─────────────────────────────────
    adapter_path = os.path.join(SFT_OUTPUT_DIR, "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved → {adapter_path}")

    # ── 9. Merge LoRA → 完整模型（GRPO 的起点） ──────────────
    print("\nMerging LoRA weights into base model...")
    # 加载最佳 adapter，merge，保存完整权重
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # merge 在 CPU 上做，避免显存不足
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(SFT_MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SFT_MERGED_DIR)
    print(f"Merged model saved → {SFT_MERGED_DIR}")

    wandb.run.summary["sft_merged_path"] = SFT_MERGED_DIR
    wandb.finish()
    print("\nSFT training complete!")


if __name__ == "__main__":
    main()

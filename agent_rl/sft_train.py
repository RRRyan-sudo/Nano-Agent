"""
Stage 1: Agent SFT 冷启动
模型起点: Qwen2.5-1.5B-Instruct
数据: glaive-function-calling-v2（5K 样本）
目标: 强化模型的工具调用格式一致性和准确性

运行方式:
    python sft_train.py

WandB 项目: agent-rl-posttrain / run: agent-sft
"""

import os
import wandb
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from config import (
    MODEL_NAME, SFT_OUTPUT_DIR, SFT_MERGED_DIR,
    WANDB_PROJECT, WANDB_ENTITY, AGENT_SFT_CONFIG,
)
from data import load_agent_sft_data

os.makedirs(SFT_OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# WandB Callback
# ──────────────────────────────────────────────────────────────

class SFTWandBCallback(TrainerCallback):
    """注入额外 WandB 指标：perplexity, tokens/sec。"""

    def __init__(self):
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        import time
        self._step_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return

        extra = {}

        # Perplexity
        if "loss" in logs:
            try:
                extra["train/perplexity"] = float(np.exp(logs["loss"]))
            except OverflowError:
                extra["train/perplexity"] = float("inf")

        if extra:
            wandb.log(extra, step=state.global_step)


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    # ── 0. WandB ─────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="agent-sft",
        config={
            "model": MODEL_NAME,
            "stage": "Agent SFT",
            "dataset": "glaive-function-calling-v2",
            **AGENT_SFT_CONFIG,
        },
    )

    # ── 1. Tokenizer ─────────────────────────────────────────
    print(f"Loading tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. 数据集 ────────────────────────────────────────────
    train_dataset = load_agent_sft_data(tokenizer)

    # ── 3. 模型 ──────────────────────────────────────────────
    print(f"Loading model from: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── 4. LoRA ──────────────────────────────────────────────
    peft_config = LoraConfig(
        r=AGENT_SFT_CONFIG["lora_r"],
        lora_alpha=AGENT_SFT_CONFIG["lora_alpha"],
        target_modules=AGENT_SFT_CONFIG["lora_target_modules"],
        lora_dropout=AGENT_SFT_CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 5. SFT 配置 ──────────────────────────────────────────
    sft_args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        max_seq_length=AGENT_SFT_CONFIG["max_seq_length"],
        packing=AGENT_SFT_CONFIG["packing"],

        # 训练
        num_train_epochs=AGENT_SFT_CONFIG["num_train_epochs"],
        per_device_train_batch_size=AGENT_SFT_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=AGENT_SFT_CONFIG["gradient_accumulation_steps"],
        learning_rate=AGENT_SFT_CONFIG["learning_rate"],
        lr_scheduler_type=AGENT_SFT_CONFIG["lr_scheduler_type"],
        warmup_ratio=AGENT_SFT_CONFIG["warmup_ratio"],

        # 精度
        bf16=AGENT_SFT_CONFIG["bf16"],
        optim=AGENT_SFT_CONFIG["optim"],
        gradient_checkpointing=AGENT_SFT_CONFIG["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # 日志 & 保存
        logging_steps=AGENT_SFT_CONFIG["logging_steps"],
        save_strategy=AGENT_SFT_CONFIG["save_strategy"],
        save_steps=AGENT_SFT_CONFIG["save_steps"],
        save_total_limit=AGENT_SFT_CONFIG["save_total_limit"],

        # WandB
        report_to="wandb",
        run_name="agent-sft",

        # 数据集文本字段
        dataset_text_field="text",
    )

    # ── 6. Trainer ───────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[SFTWandBCallback()],
    )

    # ── 7. 训练 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Agent SFT Training")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Data   : glaive-function-calling-v2 ({len(train_dataset)} samples)")
    print(f"  LoRA r : {AGENT_SFT_CONFIG['lora_r']}")
    print(f"  Epochs : {AGENT_SFT_CONFIG['num_train_epochs']}")
    print("=" * 60 + "\n")

    trainer.train()

    # ── 8. 保存 LoRA adapter ──────────────────────────────────
    adapter_path = os.path.join(SFT_OUTPUT_DIR, "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved → {adapter_path}")

    # ── 9. Merge LoRA → 完整模型 ──────────────────────────────
    print("Merging LoRA weights into base model...")
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = merged_model.merge_and_unload()

    os.makedirs(SFT_MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(SFT_MERGED_DIR)
    tokenizer.save_pretrained(SFT_MERGED_DIR)
    print(f"Merged model saved → {SFT_MERGED_DIR}")

    wandb.run.summary["sft_output_path"] = SFT_MERGED_DIR
    wandb.finish()
    print("\nAgent SFT training complete!")


if __name__ == "__main__":
    main()

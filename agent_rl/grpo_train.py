"""
Stage 2 Phase 1: Agent GRPO 单轮工具调用强化学习
模型起点: Agent SFT merged 模型 (outputs/sft_merged)
数据: 可验证的代码执行任务
Reward: 工具调用合法性 + 工具选择 + 任务完成度 - 超长/重复惩罚
目标: 通过 RL 强化工具调用的准确性和任务完成能力

运行方式:
    python grpo_train.py

WandB 项目: agent-rl-posttrain / run: agent-grpo
"""

import os
import wandb
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

from config import (
    MODEL_NAME, SFT_MERGED_DIR, GRPO_OUTPUT_DIR,
    WANDB_PROJECT, WANDB_ENTITY, AGENT_GRPO_CONFIG,
    AGENT_REWARD_WEIGHTS, AGENT_SYSTEM_PROMPT, EVAL_SAMPLE_SIZE,
)
from data import load_agent_rl_tasks, load_agent_eval_tasks
from rewards import agent_reward_fn, _reward_components_buffer
from tools import parse_tool_calls, SafeToolExecutor

os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 定期评测
# ──────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, eval_tasks: list[dict], device, step: int):
    """
    在评测任务上做 greedy 推理 + 工具执行，计算任务完成率。
    """
    model.eval()
    correct = 0
    valid_calls = 0
    rows = []

    with torch.no_grad():
        for task in eval_tasks:
            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user",   "content": task["task"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)

            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )

            # 解析并执行工具调用
            tool_calls = parse_tool_calls(response)
            has_valid_call = bool(tool_calls)
            if has_valid_call:
                valid_calls += 1

            is_correct = False
            tool_result = "N/A"
            if tool_calls:
                executor = SafeToolExecutor(timeout=5)
                try:
                    results = []
                    for tc in tool_calls:
                        result = executor.execute(tc["name"], tc["arguments"])
                        results.append({"name": tc["name"], "result": result})
                    tool_result = results[0]["result"][:100] if results else "N/A"

                    from rewards import verify_result
                    is_correct = verify_result(results, task["expected_answer"])
                finally:
                    executor.cleanup()

            if is_correct:
                correct += 1

            rows.append([
                task["task"][:60],
                task["expected_answer"],
                tool_result[:50],
                "Y" if has_valid_call else "N",
                "Y" if is_correct else "N",
            ])

    n = len(eval_tasks)
    accuracy = correct / n if n > 0 else 0
    valid_rate = valid_calls / n if n > 0 else 0

    table = wandb.Table(
        columns=["task", "expected", "tool_result", "valid_call", "correct"],
        data=rows,
    )
    wandb.log({
        "eval/accuracy": accuracy,
        "eval/valid_call_rate": valid_rate,
        "eval/predictions": table,
    }, step=step)
    print(f"[Eval @ step {step}] accuracy={accuracy:.3f} valid_calls={valid_rate:.3f} ({correct}/{n})")

    model.train()
    return accuracy


# ──────────────────────────────────────────────────────────────
# WandB Callback
# ──────────────────────────────────────────────────────────────

class AgentGRPOCallback(TrainerCallback):
    """注入 reward 各分量和定期评测。"""

    def __init__(self, eval_tasks, tokenizer, eval_steps: int):
        self.eval_tasks = eval_tasks
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        extra = {}
        if _reward_components_buffer:
            extra.update(_reward_components_buffer)
        if "kl" in logs:
            extra["train/kl"] = logs["kl"]
        if extra:
            wandb.log(extra, step=state.global_step)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if (
            state.is_world_process_zero
            and state.global_step > 0
            and state.global_step % self.eval_steps == 0
            and model is not None
        ):
            device = next(model.parameters()).device
            run_eval(model, self.tokenizer, self.eval_tasks, device, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        wandb.run.summary["train/total_steps"] = state.global_step
        if model is not None:
            device = next(model.parameters()).device
            final_acc = run_eval(
                model, self.tokenizer, self.eval_tasks, device, state.global_step
            )
            wandb.run.summary["eval/final_accuracy"] = final_acc


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    # ── 0. WandB ─────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="agent-grpo",
        config={
            "model": SFT_MERGED_DIR,
            "stage": "Agent GRPO",
            "dataset": "agent_rl_tasks",
            **AGENT_GRPO_CONFIG,
            **{f"reward/{k}": v for k, v in AGENT_REWARD_WEIGHTS.items()},
        },
    )

    # ── 1. Tokenizer ─────────────────────────────────────────
    model_path = SFT_MERGED_DIR if os.path.exists(SFT_MERGED_DIR) else MODEL_NAME
    if not os.path.exists(SFT_MERGED_DIR):
        print(f"WARNING: SFT merged model not found at {SFT_MERGED_DIR}")
        print(f"         Falling back to base model: {MODEL_NAME}")

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── 2. 数据集 ────────────────────────────────────────────
    train_dataset = load_agent_rl_tasks(tokenizer)
    eval_tasks = load_agent_eval_tasks()[:EVAL_SAMPLE_SIZE]

    # ── 3. 模型 ──────────────────────────────────────────────
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ── 4. LoRA ──────────────────────────────────────────────
    peft_config = LoraConfig(
        r=AGENT_GRPO_CONFIG["lora_r"],
        lora_alpha=AGENT_GRPO_CONFIG["lora_alpha"],
        target_modules=AGENT_GRPO_CONFIG["lora_target_modules"],
        lora_dropout=AGENT_GRPO_CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 5. GRPO 配置 ─────────────────────────────────────────
    grpo_args = GRPOConfig(
        output_dir=GRPO_OUTPUT_DIR,

        # GRPO 核心
        num_generations=AGENT_GRPO_CONFIG["num_generations"],
        max_completion_length=AGENT_GRPO_CONFIG["max_completion_length"],
        temperature=AGENT_GRPO_CONFIG["temperature"],
        beta=AGENT_GRPO_CONFIG["beta"],

        # 训练
        num_train_epochs=AGENT_GRPO_CONFIG["num_train_epochs"],
        per_device_train_batch_size=AGENT_GRPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=AGENT_GRPO_CONFIG["gradient_accumulation_steps"],
        learning_rate=AGENT_GRPO_CONFIG["learning_rate"],
        lr_scheduler_type=AGENT_GRPO_CONFIG["lr_scheduler_type"],
        warmup_ratio=AGENT_GRPO_CONFIG["warmup_ratio"],

        # 稳定性
        bf16=AGENT_GRPO_CONFIG["bf16"],
        gradient_checkpointing=AGENT_GRPO_CONFIG["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=AGENT_GRPO_CONFIG["max_grad_norm"],

        # 日志 & 保存
        logging_steps=AGENT_GRPO_CONFIG["logging_steps"],
        save_strategy=AGENT_GRPO_CONFIG["save_strategy"],
        save_steps=AGENT_GRPO_CONFIG["save_steps"],
        save_total_limit=AGENT_GRPO_CONFIG["save_total_limit"],

        # WandB
        report_to="wandb",
        run_name="agent-grpo",
    )

    # ── 6. Callback ──────────────────────────────────────────
    grpo_callback = AgentGRPOCallback(
        eval_tasks=eval_tasks,
        tokenizer=tokenizer,
        eval_steps=AGENT_GRPO_CONFIG["eval_steps"],
    )

    # ── 7. Trainer ───────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=agent_reward_fn,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[grpo_callback],
    )

    # ── 8. 训练 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Agent GRPO Training")
    print(f"  Base   : {model_path}")
    print(f"  Data   : {len(train_dataset)} agent tasks")
    print(f"  Gens   : {AGENT_GRPO_CONFIG['num_generations']} per prompt")
    rw = AGENT_REWARD_WEIGHTS
    print(f"  Reward : complete({rw['task_completion']}) + valid({rw['tool_call_valid']})"
          f" + select({rw['tool_selection']}) + overlong({rw['overlong_penalty']})")
    print(f"  β (KL) : {AGENT_GRPO_CONFIG['beta']}")
    print("=" * 60 + "\n")

    trainer.train()

    # ── 9. 保存最终模型 ──────────────────────────────────────
    final_path = os.path.join(GRPO_OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"GRPO model saved → {final_path}")

    wandb.run.summary["grpo_output_path"] = final_path
    wandb.finish()
    print("\nAgent GRPO training complete!")


if __name__ == "__main__":
    main()

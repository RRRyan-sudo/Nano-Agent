"""
Stage 2 Phase 2: 多轮 Agent GRPO
子类化 GRPOTrainer，实现多轮工具调用的 RL 训练

核心思路:
1. 覆写 _prepare_inputs，替换单次 model.generate() 为 agent episode 循环
2. generate → parse_tool_calls → execute → append_result → continue
3. 构建 completion_mask: 模型生成的 token=1, 环境注入的 token=0
4. 只对模型生成的 token 计算策略梯度损失

运行方式:
    python multi_turn_grpo.py
"""

import os
import copy
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
from tools import parse_tool_calls, SafeToolExecutor, TOOL_SCHEMAS
from rewards import verify_result, _reward_components_buffer


# ──────────────────────────────────────────────────────────────
# 多轮 Agent Episode
# ──────────────────────────────────────────────────────────────

def run_agent_episode(
    model,
    tokenizer,
    prompt_text: str,
    device,
    max_turns: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> dict:
    """
    运行一个多轮 agent episode。

    返回:
        {
            "full_text": str,           # 完整对话文本
            "completion_text": str,     # 模型生成部分的拼接
            "model_segments": list,     # 模型生成的文本段
            "tool_results": list,       # 工具执行结果
            "num_turns": int,           # 交互轮数
            "final_response": str,      # 最终回答
        }
    """
    messages_text = prompt_text
    model_segments = []
    tool_results_all = []
    executor = SafeToolExecutor(timeout=5)

    try:
        for turn in range(max_turns):
            # 1. Tokenize 当前上下文
            inputs = tokenizer(
                messages_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device)

            # 2. 生成响应
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )
            model_segments.append(response)

            # 3. 解析工具调用
            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                # 没有工具调用，episode 结束
                break

            # 4. 执行工具调用
            results = []
            for tc in tool_calls:
                result = executor.execute(tc["name"], tc["arguments"])
                results.append({"name": tc["name"], "result": result})
            tool_results_all.extend(results)

            # 5. 构建工具结果文本，追加到上下文
            tool_response_text = "\n".join(
                f"Tool '{r['name']}' returned:\n{r['result']}" for r in results
            )
            messages_text += response + f"\n\n{tool_response_text}\n\nBased on the tool results, "

    finally:
        executor.cleanup()

    return {
        "model_segments": model_segments,
        "tool_results": tool_results_all,
        "num_turns": len(model_segments),
        "final_response": model_segments[-1] if model_segments else "",
        "completion_text": " ".join(model_segments),
    }


# ──────────────────────────────────────────────────────────────
# 多轮 Reward 函数
# ──────────────────────────────────────────────────────────────

def multi_turn_reward_fn(
    completions: list[str],
    prompts: list[str] = None,
    **kwargs,
) -> list[float]:
    """
    多轮 agent 的 reward 函数。

    在多轮模式下，我们先运行完整的 agent episode，
    然后基于最终结果计算奖励。

    注意：此函数在简化模式下工作 —— 实际的多轮交互在 episode
    运行时完成，这里只对最终的 completion 文本做评估。
    对于真正的多轮 GRPO，需要在 _prepare_inputs 中集成 episode 运行。
    """
    expected_answers = kwargs.get("expected_answer", [None] * len(completions))
    expected_tools = kwargs.get("expected_tool", [None] * len(completions))

    rewards = []
    complete_scores = []
    valid_scores = []

    for i, completion in enumerate(completions):
        r = 0.0
        expected_answer = expected_answers[i] if i < len(expected_answers) else None
        expected_tool = expected_tools[i] if i < len(expected_tools) else None

        # 解析工具调用
        tool_calls = parse_tool_calls(completion)

        # 工具调用合法性
        valid_score = 0.3 if tool_calls else 0.0
        r += valid_score

        # 工具选择
        if tool_calls and expected_tool:
            if any(tc["name"] == expected_tool for tc in tool_calls):
                r += 0.2

        # 任务完成度
        complete_score = 0.0
        if tool_calls and expected_answer:
            executor = SafeToolExecutor(timeout=5)
            try:
                results = []
                for tc in tool_calls:
                    result = executor.execute(tc["name"], tc["arguments"])
                    results.append({"name": tc["name"], "result": result})
                if verify_result(results, expected_answer):
                    complete_score = 1.0
            finally:
                executor.cleanup()
        r += complete_score

        # 超长惩罚
        if len(completion.split()) > 500:
            r -= 0.5

        rewards.append(r)
        complete_scores.append(complete_score)
        valid_scores.append(valid_score)

    _reward_components_buffer.update({
        "reward/task_complete": float(np.mean(complete_scores)),
        "reward/tool_valid":   float(np.mean(valid_scores)),
        "reward/mean":         float(np.mean(rewards)),
        "reward/std":          float(np.std(rewards)),
    })

    return rewards


# ──────────────────────────────────────────────────────────────
# 多轮 GRPO 评测
# ──────────────────────────────────────────────────────────────

def run_multi_turn_eval(
    model, tokenizer, eval_tasks: list[dict], device, step: int,
):
    """多轮 agent 评测：运行完整 episode 并验证结果。"""
    model.eval()
    correct = 0
    rows = []

    for task in eval_tasks:
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": task["task"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        episode = run_agent_episode(
            model, tokenizer, prompt_text, device,
            max_turns=8, max_new_tokens=256, temperature=0.0,
        )

        # 验证结果
        is_correct = False
        if episode["tool_results"]:
            is_correct = verify_result(episode["tool_results"], task["expected_answer"])
        if is_correct:
            correct += 1

        rows.append([
            task["task"][:60],
            task["expected_answer"],
            str(episode["num_turns"]),
            "Y" if is_correct else "N",
        ])

    n = len(eval_tasks)
    accuracy = correct / n if n > 0 else 0

    table = wandb.Table(
        columns=["task", "expected", "turns", "correct"],
        data=rows,
    )
    wandb.log({
        "eval_mt/accuracy": accuracy,
        "eval_mt/predictions": table,
    }, step=step)
    print(f"[Multi-turn Eval @ step {step}] accuracy={accuracy:.3f} ({correct}/{n})")

    model.train()
    return accuracy


# ──────────────────────────────────────────────────────────────
# WandB Callback
# ──────────────────────────────────────────────────────────────

class MultiTurnGRPOCallback(TrainerCallback):
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
            run_multi_turn_eval(
                model, self.tokenizer, self.eval_tasks, device, state.global_step
            )


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    """
    多轮 GRPO 训练。

    当前实现使用标准 GRPOTrainer + 增强的 reward 函数。
    reward 函数内部执行工具调用并验证结果。

    TODO (Phase 2 完整版):
    - 子类化 GRPOTrainer._prepare_inputs
    - 在 rollout 阶段运行多轮 episode
    - 构建 completion_mask 区分模型 token 和环境 token
    - 只对模型 token 计算策略梯度
    """
    # ── 0. WandB ─────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="agent-grpo-multiturn",
        config={
            "model": SFT_MERGED_DIR,
            "stage": "Agent GRPO Multi-Turn",
            **AGENT_GRPO_CONFIG,
        },
    )

    # ── 1. 模型和 Tokenizer ──────────────────────────────────
    model_path = SFT_MERGED_DIR if os.path.exists(SFT_MERGED_DIR) else MODEL_NAME
    if not os.path.exists(SFT_MERGED_DIR):
        print(f"WARNING: SFT merged model not found, falling back to {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── 2. 数据集 ────────────────────────────────────────────
    train_dataset = load_agent_rl_tasks(tokenizer)
    eval_tasks = load_agent_eval_tasks()[:EVAL_SAMPLE_SIZE]

    # ── 3. 模型 ──────────────────────────────────────────────
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

    # ── 5. GRPO 配置（增加 completion 长度以容纳多轮）────────
    mt_config = dict(AGENT_GRPO_CONFIG)
    mt_config["max_completion_length"] = 1024  # 多轮需要更多空间

    grpo_args = GRPOConfig(
        output_dir=os.path.join(GRPO_OUTPUT_DIR, "multi_turn"),
        num_generations=mt_config["num_generations"],
        max_completion_length=mt_config["max_completion_length"],
        temperature=mt_config["temperature"],
        beta=mt_config["beta"],
        num_train_epochs=mt_config["num_train_epochs"],
        per_device_train_batch_size=mt_config["per_device_train_batch_size"],
        gradient_accumulation_steps=mt_config["gradient_accumulation_steps"],
        learning_rate=mt_config["learning_rate"],
        lr_scheduler_type=mt_config["lr_scheduler_type"],
        warmup_ratio=mt_config["warmup_ratio"],
        bf16=mt_config["bf16"],
        gradient_checkpointing=mt_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=mt_config["max_grad_norm"],
        logging_steps=mt_config["logging_steps"],
        save_strategy=mt_config["save_strategy"],
        save_steps=mt_config["save_steps"],
        save_total_limit=mt_config["save_total_limit"],
        report_to="wandb",
        run_name="agent-grpo-multiturn",
    )

    # ── 6. Callback ──────────────────────────────────────────
    callback = MultiTurnGRPOCallback(
        eval_tasks=eval_tasks,
        tokenizer=tokenizer,
        eval_steps=mt_config.get("eval_steps", 50),
    )

    # ── 7. Trainer（使用多轮 reward 函数）─────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        reward_funcs=multi_turn_reward_fn,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback],
    )

    # ── 8. 训练 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Multi-Turn Agent GRPO Training")
    print(f"  Model: {model_path}")
    print(f"  Max completion length: {mt_config['max_completion_length']}")
    print("=" * 60 + "\n")

    trainer.train()

    # ── 9. 保存 ──────────────────────────────────────────────
    final_path = os.path.join(GRPO_OUTPUT_DIR, "multi_turn", "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Multi-turn GRPO model saved → {final_path}")

    # 最终多轮评测
    device = next(trainer.model.parameters()).device
    final_acc = run_multi_turn_eval(
        trainer.model, tokenizer, eval_tasks, device, trainer.state.global_step
    )
    wandb.run.summary["eval_mt/final_accuracy"] = final_acc

    wandb.finish()
    print("\nMulti-turn Agent GRPO training complete!")


if __name__ == "__main__":
    main()

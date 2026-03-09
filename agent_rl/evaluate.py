"""
Agent 能力评测
对比三阶段模型: Base Instruct → Agent SFT → Agent GRPO

评测维度:
1. 工具调用准确率 (tool_call_valid_rate)
2. 工具选择正确率 (tool_selection_rate)
3. 任务完成率 (task_completion_rate)
4. 平均响应长度 (avg_response_length)

运行方式:
    python evaluate.py
"""

import os
import json
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_NAME, SFT_MERGED_DIR, GRPO_OUTPUT_DIR,
    WANDB_PROJECT, WANDB_ENTITY, AGENT_SYSTEM_PROMPT, EVAL_SAMPLE_SIZE,
)
from data import load_agent_eval_tasks
from tools import parse_tool_calls, SafeToolExecutor
from rewards import verify_result


def evaluate_model(
    model_path: str,
    model_name: str,
    eval_tasks: list[dict],
    tokenizer=None,
) -> dict:
    """
    评测一个模型在 agent 任务上的表现。
    返回评测结果字典。
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Model path: {model_path}")
    print(f"  Tasks: {len(eval_tasks)}")
    print(f"{'='*60}\n")

    # 加载模型
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    # 评测
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "total_tasks": len(eval_tasks),
        "valid_calls": 0,
        "correct_tools": 0,
        "completed_tasks": 0,
        "total_response_length": 0,
        "details": [],
    }

    with torch.no_grad():
        for i, task in enumerate(eval_tasks):
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

            resp_len = len(response.split())
            results["total_response_length"] += resp_len

            # 解析工具调用
            tool_calls = parse_tool_calls(response)
            has_valid_call = bool(tool_calls)
            correct_tool = False
            task_completed = False
            tool_result = ""

            if has_valid_call:
                results["valid_calls"] += 1

                # 检查工具选择
                if any(tc["name"] == task["expected_tool"] for tc in tool_calls):
                    correct_tool = True
                    results["correct_tools"] += 1

                # 执行并验证
                executor = SafeToolExecutor(timeout=5)
                try:
                    exec_results = []
                    for tc in tool_calls:
                        result = executor.execute(tc["name"], tc["arguments"])
                        exec_results.append({"name": tc["name"], "result": result})
                    tool_result = exec_results[0]["result"] if exec_results else ""

                    if verify_result(exec_results, task["expected_answer"]):
                        task_completed = True
                        results["completed_tasks"] += 1
                finally:
                    executor.cleanup()

            detail = {
                "task": task["task"],
                "expected_answer": task["expected_answer"],
                "expected_tool": task["expected_tool"],
                "response_preview": response[:200],
                "has_valid_call": has_valid_call,
                "correct_tool": correct_tool,
                "task_completed": task_completed,
                "tool_result_preview": tool_result[:100],
                "response_length": resp_len,
            }
            results["details"].append(detail)

            status = "PASS" if task_completed else ("CALL" if has_valid_call else "FAIL")
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(eval_tasks)}] {status} | {task['task'][:50]}...")

    # 计算汇总指标
    n = results["total_tasks"]
    results["metrics"] = {
        "tool_call_valid_rate":  results["valid_calls"] / n,
        "tool_selection_rate":   results["correct_tools"] / n,
        "task_completion_rate":  results["completed_tasks"] / n,
        "avg_response_length":   results["total_response_length"] / n,
    }

    m = results["metrics"]
    print(f"\n  Results for {model_name}:")
    print(f"    Tool call valid rate:  {m['tool_call_valid_rate']:.3f}")
    print(f"    Tool selection rate:   {m['tool_selection_rate']:.3f}")
    print(f"    Task completion rate:  {m['task_completion_rate']:.3f}")
    print(f"    Avg response length:   {m['avg_response_length']:.1f} words")

    # 释放显存
    del model
    torch.cuda.empty_cache()

    return results


def main():
    # ── 0. WandB ─────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="agent-eval",
        config={"stage": "Evaluation"},
    )

    # ── 1. 加载评测任务 ──────────────────────────────────────
    eval_tasks = load_agent_eval_tasks()[:EVAL_SAMPLE_SIZE]
    print(f"Loaded {len(eval_tasks)} evaluation tasks")

    # ── 2. 确定要评测的模型 ──────────────────────────────────
    models_to_eval = [
        (MODEL_NAME, "Base Instruct"),
    ]
    if os.path.exists(SFT_MERGED_DIR):
        models_to_eval.append((SFT_MERGED_DIR, "Agent SFT"))
    else:
        print(f"SKIP Agent SFT: {SFT_MERGED_DIR} not found")

    grpo_final = os.path.join(GRPO_OUTPUT_DIR, "final")
    if os.path.exists(grpo_final):
        models_to_eval.append((grpo_final, "Agent GRPO"))
    else:
        print(f"SKIP Agent GRPO: {grpo_final} not found")

    # ── 3. 逐个评测 ─────────────────────────────────────────
    all_results = []
    for model_path, model_name in models_to_eval:
        results = evaluate_model(model_path, model_name, eval_tasks)
        all_results.append(results)

    # ── 4. 汇总对比 ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    # 表格
    header = f"{'Model':<20} {'Valid Call':>12} {'Correct Tool':>14} {'Completion':>12} {'Avg Len':>10}"
    print(header)
    print("-" * 70)

    table_data = []
    for r in all_results:
        m = r["metrics"]
        row = f"{r['model_name']:<20} {m['tool_call_valid_rate']:>11.1%} {m['tool_selection_rate']:>13.1%} {m['task_completion_rate']:>11.1%} {m['avg_response_length']:>9.1f}"
        print(row)
        table_data.append([
            r["model_name"],
            f"{m['tool_call_valid_rate']:.1%}",
            f"{m['tool_selection_rate']:.1%}",
            f"{m['task_completion_rate']:.1%}",
            f"{m['avg_response_length']:.1f}",
        ])

    # WandB 表格
    comparison_table = wandb.Table(
        columns=["Model", "Valid Call Rate", "Correct Tool Rate", "Completion Rate", "Avg Length"],
        data=table_data,
    )
    wandb.log({"eval/comparison": comparison_table})

    # WandB summary
    for r in all_results:
        prefix = r["model_name"].lower().replace(" ", "_")
        for k, v in r["metrics"].items():
            wandb.run.summary[f"{prefix}/{k}"] = v

    # ── 5. 保存详细结果 ──────────────────────────────────────
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    serializable = []
    for r in all_results:
        serializable.append({
            "model_name": r["model_name"],
            "model_path": r["model_path"],
            "total_tasks": r["total_tasks"],
            "metrics": r["metrics"],
            "details": r["details"],
        })
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved → {output_file}")

    wandb.finish()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

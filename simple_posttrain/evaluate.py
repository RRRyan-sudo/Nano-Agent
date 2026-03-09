"""
Pipeline 评估脚本：对比三个阶段的 GSM8K 表现
    Base Model  → SFT Model → GRPO Model

用法:
    python evaluate.py [--n_samples 100]

输出:
    - 终端打印对比表格
    - eval_results.json（便于 blog 引用）
    - WandB Summary + Table（可截图用于 blog）
"""

import os
import re
import json
import argparse
import wandb
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import (
    MODEL_NAME, SFT_MERGED_DIR, GRPO_OUTPUT_DIR,
    WANDB_PROJECT, WANDB_ENTITY, MATH_SYSTEM_PROMPT,
)


# ──────────────────────────────────────────────────────────────
# 复用 grpo_train.py 中的 answer 提取逻辑（独立版本，避免循环导入）
# ──────────────────────────────────────────────────────────────

def normalize_number(s: str) -> str:
    return s.replace(",", "").replace(" ", "").strip()


def extract_answer(text: str) -> str | None:
    m = re.search(r"<answer>\s*([\d,\.]+)", text)
    if m:
        return normalize_number(m.group(1))
    m = re.search(r"####\s*([\d,\.]+)", text)
    if m:
        return normalize_number(m.group(1))
    nums = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)
    return normalize_number(nums[-1]) if nums else None


def extract_gt_answer(answer_str: str) -> str | None:
    m = re.search(r"####\s*([\d,\.]+)", answer_str)
    return normalize_number(m.group(1)) if m else None


def has_reasoning_format(text: str) -> bool:
    has_r = bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL))
    has_a = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
    return has_r and has_a


# ──────────────────────────────────────────────────────────────
# 模型加载工具
# ──────────────────────────────────────────────────────────────

def load_model(model_path: str, adapter_path: str | None = None):
    """通用模型加载：支持完整模型和 LoRA adapter。"""
    print(f"  Loading: {model_path}" + (f" + adapter {adapter_path}" if adapter_path else ""))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────────────────────
# 单条推理
# ──────────────────────────────────────────────────────────────

def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 300) -> str:
    messages = [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────
# 评估单个模型
# ──────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, dataset, label: str) -> dict:
    """在 dataset 上跑完整评测，返回指标字典。"""
    print(f"\n{'─'*50}")
    print(f"Evaluating: {label}  ({len(dataset)} samples)")
    print(f"{'─'*50}")

    correct       = 0
    format_ok     = 0
    total_len     = 0
    rows          = []

    for i, sample in enumerate(dataset):
        response = generate_answer(model, tokenizer, sample["question"])
        gt       = extract_gt_answer(sample["answer"])
        pred     = extract_answer(response)
        is_correct  = (pred == gt) if (pred and gt) else False
        is_format   = has_reasoning_format(response)

        if is_correct:  correct   += 1
        if is_format:   format_ok += 1
        total_len += len(response)

        rows.append({
            "question":     sample["question"][:100],
            "gt_answer":    gt or "?",
            "pred_answer":  pred or "N/A",
            "correct":      is_correct,
            "format_ok":    is_format,
            "response_len": len(response),
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(dataset)}  running acc={correct/(i+1):.3f}")

    n = len(dataset)
    metrics = {
        "accuracy":             correct   / n,
        "format_compliance":    format_ok / n,
        "avg_response_length":  total_len / n,
        "n_correct":            correct,
        "n_format_ok":          format_ok,
        "n_samples":            n,
    }
    print(f"  Result → accuracy={metrics['accuracy']:.3f}  format={metrics['format_compliance']:.3f}")
    return metrics, rows


# ──────────────────────────────────────────────────────────────
# 打印对比表格
# ──────────────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    labels = list(results.keys())
    print("\n" + "="*70)
    print("  EVALUATION RESULTS COMPARISON")
    print("="*70)
    header = f"{'Metric':<28}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    print("─"*70)

    metrics_to_show = [
        ("Accuracy",          "accuracy",             ".3f"),
        ("Format Compliance", "format_compliance",    ".3f"),
        ("Avg Response Len",  "avg_response_length",  ".0f"),
        ("N Correct",         "n_correct",            "d"),
    ]
    for name, key, fmt in metrics_to_show:
        row = f"{name:<28}"
        for label in labels:
            val = results[label]["metrics"][key]
            row += f"{val:{'>14' if fmt != 'd' else '>14'}{fmt}}"
        print(row)
    print("="*70)


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of GSM8K test samples to evaluate")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip WandB logging")
    args = parser.parse_args()

    # ── WandB ─────────────────────────────────────────────────
    if not args.no_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="pipeline-evaluation",
            config={"n_samples": args.n_samples},
        )

    # ── 数据集 ────────────────────────────────────────────────
    print(f"Loading GSM8K test[:{ args.n_samples}]...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(min(args.n_samples, len(dataset))))
    print(f"Evaluation set: {len(dataset)} samples")

    # ── 定义要评测的模型 ─────────────────────────────────────
    grpo_final = os.path.join(GRPO_OUTPUT_DIR, "final")

    model_configs = [
        # (label,          model_path,     adapter_path)
        ("1_Base",        MODEL_NAME,     None),
        ("2_SFT",         SFT_MERGED_DIR, None),
        ("3_GRPO",        grpo_final,     None),
    ]

    # 跳过不存在的模型（允许只评测部分阶段）
    valid_configs = []
    for label, mpath, apath in model_configs:
        if label == "1_Base" or os.path.exists(mpath):
            valid_configs.append((label, mpath, apath))
        else:
            print(f"[SKIP] {label}: path not found ({mpath})")

    # ── 逐模型评测 ────────────────────────────────────────────
    all_results = {}

    for label, model_path, adapter_path in valid_configs:
        print(f"\n{'='*60}")
        print(f"  Stage: {label}")
        model, tokenizer = load_model(model_path, adapter_path)
        metrics, rows    = evaluate_model(model, tokenizer, dataset, label)
        all_results[label] = {"metrics": metrics, "rows": rows}

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # ── 打印对比表格 ──────────────────────────────────────────
    print_comparison_table(all_results)

    # ── 保存 JSON ─────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    json_output = {
        label: {
            "metrics": data["metrics"],
            "sample_predictions": data["rows"][:5],   # 只保留前 5 条示例
        }
        for label, data in all_results.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {output_path}")

    # ── WandB 上传 ────────────────────────────────────────────
    if not args.no_wandb:
        # Summary metrics（可在 WandB 首页直接看到）
        for label, data in all_results.items():
            for k, v in data["metrics"].items():
                if isinstance(v, (int, float)):
                    wandb.run.summary[f"{label}/{k}"] = v

        # Bar chart：各阶段准确率
        acc_table = wandb.Table(
            columns=["stage", "accuracy", "format_compliance"],
            data=[
                [label, d["metrics"]["accuracy"], d["metrics"]["format_compliance"]]
                for label, d in all_results.items()
            ],
        )
        wandb.log({
            "eval/accuracy_comparison":          wandb.plot.bar(acc_table, "stage", "accuracy",          title="Accuracy by Stage"),
            "eval/format_compliance_comparison": wandb.plot.bar(acc_table, "stage", "format_compliance", title="Format Compliance by Stage"),
        })

        # Prediction Table for each stage
        for label, data in all_results.items():
            wt = wandb.Table(
                columns=["question", "gt", "pred", "correct", "format_ok"],
                data=[
                    [r["question"], r["gt_answer"], r["pred_answer"],
                     "✓" if r["correct"] else "✗",
                     "✓" if r["format_ok"] else "✗"]
                    for r in data["rows"]
                ],
            )
            wandb.log({f"predictions/{label}": wt})

        wandb.finish()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

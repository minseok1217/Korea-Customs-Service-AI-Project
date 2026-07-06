#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = Path(
    "/data/2_data_server/cv-07/dice/the Korea Customs Service/project/model/text_classification/qwen3_4b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen prompt-based text classification.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--labels", required=True, help="Comma-separated class labels.")
    parser.add_argument("--text", default=None, help="Single or multi-item text to classify.")
    parser.add_argument("--input-csv", type=Path, default=None, help="CSV with a text column.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def parse_labels(raw: str) -> list[str]:
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one label")
    return labels


def build_prompt(text: str, labels: list[str]) -> str:
    labels_text = ", ".join(labels)
    examples = [
        ("handcuff", {"수갑": 1}),
        ("water bottle, handcuff", {"수갑": 1}),
        ("portable lithium ion power bank, USB-C charger", {"보조배터리": 1, "USB": 1}),
        ("kitchen knife and stainless steel blade set", {"칼": 1}),
        ("e-cigarette device with refill liquid pods", {"전자담배": 1, "전자담배 액상": 1}),
        ("smartphone, tablet PC, laptop computer", {"스마트폰": 1, "태블릿PC": 1, "노트북": 1}),
        ("fireworks and small pyrotechnic items", {"폭죽": 1}),
        ("screwdriver, pliers, and adjustable wrench", {"드라이버": 1, "펜치": 1, "스패너": 1}),
        ("bullets for sporting rifle", {"탄환": 1}),
    ]
    examples_text = "\n".join(
        f'Input: "{example_text}"\nOutput: {json.dumps({"counts": counts}, ensure_ascii=False)}'
        for example_text, counts in examples
    )
    return (
        "You are a strict customs declaration text classification system.\n"
        "Classify detailed declared item names into the allowed Korean labels.\n"
        "The input may contain one item, multiple item names, OCR text, or a short product description.\n"
        "If multiple items are present, classify each relevant item and aggregate counts by label.\n"
        "Ignore items that do not belong to any allowed label.\n"
        "Use semantic meaning, not exact string matching.\n"
        f"Allowed labels: {labels_text}\n"
        "Return only a compact JSON object with key \"counts\".\n"
        "The counts object maps allowed label strings to integer counts.\n"
        "Do not include explanations, markdown, XML tags, or <think> text.\n\n"
        "Examples:\n"
        f"{examples_text}\n\n"
        f"Text:\n{text}\n"
    )


def extract_json(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        counts = {}
        for line in cleaned.splitlines():
            match = re.match(r"\s*([^:：]+)\s*[:：]\s*(\d+)\s*$", line)
            if match:
                counts[match.group(1).strip()] = int(match.group(2))
        return {"counts": counts, "raw_text": cleaned}
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return {"counts": {}, "raw_text": cleaned}
    if "counts" not in parsed and "label" in parsed:
        parsed = {"counts": {str(parsed["label"]): 1}, **parsed}
    if "counts" not in parsed:
        parsed["counts"] = {}
    return parsed


def load_inputs(args: argparse.Namespace) -> list[dict]:
    if args.text is not None:
        return [{"id": "0", "text": args.text}]
    if args.input_csv is None:
        raise ValueError("Pass either --text or --input-csv")
    rows = []
    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if args.text_column not in row:
                raise KeyError(f"text column not found: {args.text_column}")
            rows.append({"id": row.get("id", str(idx)), "text": row[args.text_column], "row": row})
    return rows


def main() -> None:
    args = parse_args()
    labels = parse_labels(args.labels)
    inputs = load_inputs(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for item in inputs:
        prompt = build_prompt(item["text"], labels)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            rendered_prompt = prompt
        encoded = tokenizer([rendered_prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][encoded.input_ids.shape[-1] :]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        pred = extract_json(decoded)
        result = {"id": item["id"], "text": item["text"], "raw": decoded, **pred}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

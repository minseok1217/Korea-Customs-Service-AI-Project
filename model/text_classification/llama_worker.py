#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT = Path("/data/2_data_server/cv-07/dice/the Korea Customs Service/project")
SCRIPT_DIR = PROJECT / "model/text_classification/qwen_text_classification"
MODEL_DIR = Path(
    os.environ.get(
        "LLAMA_MODEL_DIR",
        str(PROJECT / "model/text_classification/llama4_scout_17b_16e_instruct"),
    )
)
HOST = "127.0.0.1"
PORT = int(os.environ.get("LLAMA_WORKER_PORT", "18083"))
DEVICE_MAP = os.environ.get("LLAMA_DEVICE_MAP", "auto")
MAX_NEW_TOKENS = int(os.environ.get("LLAMA_MAX_NEW_TOKENS", "256"))

sys.path.insert(0, str(SCRIPT_DIR))
from classify_text import build_prompt, extract_json, parse_labels  # noqa: E402


if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Llama model directory not found: {MODEL_DIR}. "
        "Download or copy the model there, or set LLAMA_MODEL_DIR."
    )

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map=DEVICE_MAP,
    trust_remote_code=True,
)
MODEL.eval()


def classify(text: str, labels: list[str]) -> dict:
    prompt = build_prompt(text, labels)
    messages = [{"role": "user", "content": prompt}]
    if hasattr(TOKENIZER, "apply_chat_template"):
        rendered = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = prompt
    encoded = TOKENIZER([rendered], return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        output_ids = MODEL.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    generated = output_ids[0][encoded.input_ids.shape[-1] :]
    decoded = TOKENIZER.decode(generated, skip_special_tokens=True).strip()
    pred = extract_json(decoded)
    return {"text": text, "raw": decoded, "model": "Llama", **pred}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            device = getattr(MODEL, "device", DEVICE_MAP)
            self._send(200, {"ok": True, "model": str(MODEL_DIR), "device": str(device)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/classify":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            labels = parse_labels(payload["labels"])
            self._send(200, classify(str(payload.get("text", "")), labels))
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)


def main() -> None:
    print(f"Llama worker ready on http://{HOST}:{PORT}", flush=True)
    print(f"MODEL_DIR={MODEL_DIR}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

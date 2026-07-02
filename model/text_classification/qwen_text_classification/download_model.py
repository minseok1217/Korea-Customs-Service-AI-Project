#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot into project/model.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/2_data_server/cv-07/dice/the Korea Customs Service/project/model/qwen3_4b"),
    )
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
    )
    print(path)


if __name__ == "__main__":
    main()

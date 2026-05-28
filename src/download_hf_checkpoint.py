from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot to a local checkpoint directory.")
    parser.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None, help="HF token, if needed. You can also use HF_TOKEN env var.")
    args = parser.parse_args()

    out = snapshot_download(
        repo_id=args.model,
        local_dir=str(Path(args.output_dir).expanduser()),
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=args.token,
    )
    print(out)


if __name__ == "__main__":
    main()

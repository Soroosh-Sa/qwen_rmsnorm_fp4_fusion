#!/usr/bin/env python3
"""Collect deterministic outputs from an OpenAI-compatible TensorRT-LLM server.

This is a correctness/quality sanity tool, not a throughput benchmark.  It sends
exactly the same prompts to each engine and stores the generated text so the
outputs can be compared later.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

_TOKENIZER = None
_TOKENIZER_PATH = None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise ValueError(f"{path}:{line_no} missing required key 'prompt'")
            obj.setdefault("id", f"prompt_{line_no}")
            obj.setdefault("expected_contains", [])
            rows.append(obj)
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def _get_tokenizer(tokenizer_path: str | None):
    global _TOKENIZER, _TOKENIZER_PATH
    if not tokenizer_path:
        return None
    if _TOKENIZER is not None and _TOKENIZER_PATH == tokenizer_path:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
        )
        _TOKENIZER_PATH = tokenizer_path
        return _TOKENIZER
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        print(f"[WARN] Could not load tokenizer from {tokenizer_path}: {exc!r}", flush=True)
        return None


def _apply_chat_template(prompt: str, tokenizer_path: str | None) -> str:
    tokenizer = _get_tokenizer(tokenizer_path)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:  # pragma: no cover - tokenizer dependent
        print(f"[WARN] apply_chat_template failed: {exc!r}", flush=True)
        return prompt


def _extract_response_text(obj: Dict[str, Any]) -> Tuple[str, int | None, str | None]:
    text = ""
    completion_tokens = None
    finish_reason = None
    usage = obj.get("usage")
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

    choices = obj.get("choices") or []
    if isinstance(choices, dict):
        choices = [choices]
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason") or finish_reason
        for key in ("text", "content", "output_text", "generated_text"):
            val = choice.get(key)
            if isinstance(val, str):
                text += val
        msg = choice.get("message")
        if isinstance(msg, dict):
            for key in ("content", "text"):
                val = msg.get(key)
                if isinstance(val, str):
                    text += val
    for key in ("text", "content", "output_text", "generated_text"):
        val = obj.get(key)
        if isinstance(val, str):
            text += val
    return text, completion_tokens, finish_reason


def _count_tokens(text: str, tokenizer_path: str | None) -> int:
    tok = _get_tokenizer(tokenizer_path)
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return 0 if not text else max(1, len(text.split()))


def _request_once(
    url: str,
    model: str,
    prompt: str,
    api_mode: str,
    max_tokens: int,
    timeout_s: float,
    tokenizer_path: str | None,
    use_chat_template_for_completion: bool,
) -> Dict[str, Any]:
    api_mode = "completion" if api_mode in {"completion", "completions"} else "chat"
    if api_mode == "completion":
        prompt_to_send = _apply_chat_template(prompt, tokenizer_path) if use_chat_template_for_completion else prompt
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt_to_send,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        endpoint = "completions"
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        endpoint = "chat/completions"

    start = time.perf_counter()
    r = requests.post(f"{url.rstrip('/')}/v1/{endpoint}", json=payload, timeout=timeout_s)
    latency_s = time.perf_counter() - start
    if r.status_code >= 400:
        return {
            "success": False,
            "status_code": r.status_code,
            "error": r.text[:2000],
            "text": "",
            "completion_tokens": 0,
            "finish_reason": None,
            "latency_s": latency_s,
        }
    obj = r.json()
    text, completion_tokens, finish_reason = _extract_response_text(obj)
    if completion_tokens is None:
        completion_tokens = _count_tokens(text, tokenizer_path)
    return {
        "success": bool(text) or completion_tokens > 0,
        "status_code": r.status_code,
        "error": "" if text or completion_tokens > 0 else "empty_output",
        "text": text,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "latency_s": latency_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True, help="Label: normal, folded_base, plugin, etc.")
    parser.add_argument("--prompts", default="data/quality_prompts.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-mode", default=os.environ.get("OPENAI_API_MODE", "completion"), choices=["chat", "completion", "completions"])
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("QUALITY_MAX_NEW_TOKENS", "128")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("QUALITY_TIMEOUT_S", "300")))
    parser.add_argument("--tokenizer-path", default=os.environ.get("TOKENIZER_PATH") or os.environ.get("PLAN_MODEL") or os.environ.get("MODEL_PATH"))
    parser.add_argument("--use-chat-template-for-completion", type=int, default=int(os.environ.get("COMPLETION_USE_CHAT_TEMPLATE", "1")))
    args = parser.parse_args()

    prompts = _load_jsonl(args.prompts)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"

    # Replace output file for this target so repeated quality runs do not append stale rows.
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(prompts):
            prompt_id = str(row.get("id") or f"prompt_{i}")
            print(f"[{args.target}] prompt {i+1}/{len(prompts)}: {prompt_id}", flush=True)
            result = _request_once(
                base_url,
                args.model,
                row["prompt"],
                args.api_mode,
                args.max_tokens,
                args.timeout_s,
                args.tokenizer_path,
                bool(args.use_chat_template_for_completion),
            )
            record = {
                "target": args.target,
                "model": args.model,
                "prompt_id": prompt_id,
                "prompt": row["prompt"],
                "expected_contains": row.get("expected_contains", []),
                **result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

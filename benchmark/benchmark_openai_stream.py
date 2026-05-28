import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Optional, Tuple

import requests

_TOKENIZER = None
_TOKENIZER_PATH = None


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _debug_print(msg: str) -> None:
    if _env_flag("DEBUG_BENCHMARK_REQUESTS", "1"):
        print(msg, flush=True)


def gpu_snapshot():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True)
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {
                        "gpu_index": parts[0],
                        "gpu_name": parts[1],
                        "mem_used_mb": float(parts[2]),
                        "mem_total_mb": float(parts[3]),
                        "gpu_util_percent": float(parts[4]),
                    }
                )
        return rows
    except Exception as e:
        return [{"error": repr(e)}]


def sum_gpu_mem_gb(snapshot):
    total_mb = 0.0
    for row in snapshot:
        if "mem_used_mb" in row:
            total_mb += row["mem_used_mb"]
    return total_mb / 1024.0


def mean_gpu_util(snapshot):
    vals = [row["gpu_util_percent"] for row in snapshot if "gpu_util_percent" in row]
    return statistics.mean(vals) if vals else None


def detect_server_log_stall(required_tokens: int) -> str:
    """Detect TensorRT-LLM long-context stalls from the server log.

    Returns an empty string if no known stall pattern is seen. This prevents
    spending 15-60 minutes waiting for a request that the server accepted with
    HTTP 200 but cannot schedule because the KV-cache window is too small.
    """
    import re
    server_log = os.environ.get("SERVER_LOG", "")
    if not server_log or not os.path.exists(server_log):
        return ""
    try:
        text = open(server_log, errors="ignore").read()[-200000:]
    except Exception:
        return ""

    windows = [int(x) for x in re.findall(r"window size=(\d+)", text)]
    if windows:
        w = windows[-1]
        if w < required_tokens:
            return f"kv_cache_window_too_small_window_{w}_required_{required_tokens}"

    m = re.search(r"default_max_tokens\s*\((-?\d+)\).*?max_seq_len\s*\((\d+)\).*?splited_prompt_len\s*\((\d+)\)", text, re.S)
    if m:
        default_max = int(m.group(1))
        effective_max = int(m.group(2))
        split_len = int(m.group(3))
        if default_max < 0:
            return f"trtllm_default_max_tokens_negative_effective_{effective_max}_prompt_{split_len}"

    return ""


def _get_tokenizer(tokenizer_path: Optional[str]):
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
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from {tokenizer_path}: {repr(e)}")
        return None


def _encode_len(tokenizer, text: str) -> int:
    if tokenizer is None:
        return -1
    return len(tokenizer.encode(text, add_special_tokens=False))



def _load_prompt_template(profile: str):
    """Load a prompt template from data/assignment_prompts.jsonl.

    The JSONL file keeps assignment workload prompts separate from code so the
    same benchmark engine can run chat, code-generation, sustained-throughput,
    and long-context scenarios.
    """
    prompt_file = os.environ.get("PROMPT_FILE", "data/assignment_prompts.jsonl")
    fallback = {
        "prompt_profile": "synthetic_code_context",
        "workload_type": "long_context",
        "system_instruction": "You are a coding assistant. Read the following synthetic Python code context. Answer only the final question.",
        "context_unit": (
            "def transform_value(x):\n"
            "    y = x + 1\n"
            "    z = y * 2\n"
            "    if z % 3 == 0:\n"
            "        return z - 1\n"
            "    return z + 1\n\n"
        ),
        "final_instruction": "Question: In one sentence, summarize what transform_value repeatedly does.",
    }
    try:
        with open(prompt_file, errors="ignore") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                obj=json.loads(line)
                if obj.get("prompt_profile") == profile:
                    return obj
    except Exception as e:
        print(f"[WARN] Could not load prompt profile {profile!r} from {prompt_file}: {e!r}", flush=True)
    return fallback

def make_prompt(context_len: int) -> Tuple[str, int, int]:
    """Create a tokenizer-aware prompt for the selected assignment workload.

    Select the prompt via PROMPT_PROFILE and PROMPT_FILE. The prompt is expanded
    to fit the requested context window while leaving PROMPT_TOKEN_RESERVE tokens
    for chat-template overhead and generated output.
    """
    tokenizer_path = (
        os.environ.get("TOKENIZER_PATH")
        or os.environ.get("PLAN_MODEL")
        or os.environ.get("MODEL_PATH")
    )
    reserve_tokens = int(os.environ.get("PROMPT_TOKEN_RESERVE", "1024"))
    target_prompt_tokens = max(32, int(context_len) - reserve_tokens)

    profile = os.environ.get("PROMPT_PROFILE", "synthetic_code_context")
    template = _load_prompt_template(profile)
    header = (template.get("system_instruction") or "You are a helpful assistant.").strip() + "\n\n"
    unit = template.get("context_unit") or "Synthetic context line for benchmarking.\n"
    footer = "\n" + (template.get("final_instruction") or "Answer the final question briefly.").strip()

    tokenizer = _get_tokenizer(tokenizer_path)

    if tokenizer is None:
        # Conservative fallback: use fewer chars/token to avoid token overrun.
        target_chars = max(1, target_prompt_tokens) * 2
        body = unit * max(1, target_chars // max(1, len(unit)))
        prompt = header + body + footer
        return prompt, -1, target_prompt_tokens

    fixed = header + footer
    fixed_ids = tokenizer.encode(fixed, add_special_tokens=False)
    unit_ids = tokenizer.encode(unit, add_special_tokens=False)
    if not unit_ids:
        unit_ids = tokenizer.encode("Synthetic benchmark context.\n", add_special_tokens=False)
    remaining = max(1, target_prompt_tokens - len(fixed_ids))

    repeated_ids = []
    while len(repeated_ids) < remaining:
        repeated_ids.extend(unit_ids)
    repeated_ids = repeated_ids[:remaining]

    body = tokenizer.decode(repeated_ids, skip_special_tokens=True)
    prompt = header + body + footer
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(ids) > target_prompt_tokens:
        ids = ids[:target_prompt_tokens]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)

    return prompt, len(ids), target_prompt_tokens



def apply_chat_template_for_completion(prompt: str) -> Tuple[str, int]:
    """Optionally wrap a raw user prompt with the model's chat template for /v1/completions.

    TensorRT-LLM's /v1/completions endpoint expects a plain text prompt. For
    instruct/chat checkpoints such as Qwen3-Coder-Instruct, sending the raw user
    text can produce an immediate EOS or an empty completion. Using the tokenizer
    chat template creates the same style of prompt that /v1/chat/completions
    would normally build internally, while still avoiding the OpenAI chat server
    path that had long-context issues for 64K.
    """
    if not _env_flag("COMPLETION_USE_CHAT_TEMPLATE", "1"):
        tokenizer_path = os.environ.get("TOKENIZER_PATH") or os.environ.get("PLAN_MODEL") or os.environ.get("MODEL_PATH")
        tokenizer = _get_tokenizer(tokenizer_path)
        return prompt, _encode_len(tokenizer, prompt)

    tokenizer_path = os.environ.get("TOKENIZER_PATH") or os.environ.get("PLAN_MODEL") or os.environ.get("MODEL_PATH")
    tokenizer = _get_tokenizer(tokenizer_path)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt, _encode_len(tokenizer, prompt)

    try:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return templated, _encode_len(tokenizer, templated)
    except Exception as e:
        print(f"[WARN] Failed to apply completion chat template: {repr(e)}")
        return prompt, _encode_len(tokenizer, prompt)


def _extract_text_and_usage_from_response(obj, api_mode):
    """Return (text, completion_tokens, prompt_tokens) from OpenAI-style JSON/SSE.

    TensorRT-LLM versions differ in how /v1/completions streams data. Some use
    choices[].text, some use chat-like delta.content, and some provide a full
    JSON object even when stream=True. This extractor is intentionally tolerant.
    """
    text = ""
    completion_tokens = None
    prompt_tokens = None

    if not isinstance(obj, dict):
        return text, completion_tokens, prompt_tokens

    usage = obj.get("usage")
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens", completion_tokens)
        # Some OpenAI-compatible servers use output_tokens.
        completion_tokens = usage.get("output_tokens", completion_tokens)
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        prompt_tokens = usage.get("input_tokens", prompt_tokens)

    # A few servers put generated text at top-level.
    for key in ("text", "content", "output_text", "generated_text"):
        val = obj.get(key)
        if isinstance(val, str):
            text += val

    choices = obj.get("choices", [])
    if isinstance(choices, dict):
        choices = [choices]
    for choice in choices if isinstance(choices, list) else []:
        if not isinstance(choice, dict):
            continue
        for key in ("text", "content", "output_text", "generated_text"):
            val = choice.get(key)
            if isinstance(val, str):
                text += val

        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            for key in ("content", "text", "output_text"):
                val = delta.get(key)
                if isinstance(val, str):
                    text += val

        message = choice.get("message") or {}
        if isinstance(message, dict):
            for key in ("content", "text"):
                val = message.get(key)
                if isinstance(val, str):
                    text += val

        # Some APIs return token object/list. Only use obvious string payloads.
        token = choice.get("token")
        if isinstance(token, str):
            text += token
        elif isinstance(token, dict):
            for key in ("text", "content"):
                val = token.get(key)
                if isinstance(val, str):
                    text += val

    return text, completion_tokens, prompt_tokens

def _count_output_tokens_fallback(output_text: str, tokenizer_path: Optional[str]) -> int:
    if not output_text:
        return 0
    tokenizer = _get_tokenizer(tokenizer_path)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(output_text, add_special_tokens=False))
        except Exception:
            pass
    # Last fallback: word count, at least 1 if non-empty.
    return max(1, len(output_text.split()))


def run_one_request(url, model, context_len, max_tokens, request_id, timeout_s, api_mode):
    prompt_build_start = time.perf_counter()
    _debug_print(f"[request {request_id}] Building prompt for context={context_len}...")
    prompt, prompt_tokens_est, target_prompt_tokens = make_prompt(context_len)
    prompt_build_s = time.perf_counter() - prompt_build_start
    _debug_print(
        f"[request {request_id}] Prompt built in {prompt_build_s:.2f}s: "
        f"target_prompt_tokens={target_prompt_tokens}, "
        f"prompt_tokens_est={prompt_tokens_est}, prompt_chars={len(prompt)}"
    )
    api_mode = (api_mode or "chat").strip().lower()
    if api_mode == "completions":
        api_mode = "completion"

    if api_mode == "completion":
        prompt, templated_len = apply_chat_template_for_completion(prompt)
        if templated_len not in (None, -1):
            prompt_tokens_est = templated_len
        _debug_print(
            f"[request {request_id}] Completion prompt prepared: "
            f"use_chat_template={_env_flag('COMPLETION_USE_CHAT_TEMPLATE', '1')}, "
            f"prompt_tokens_est={prompt_tokens_est}, prompt_chars={len(prompt)}"
        )

        # /v1/completions maps max_tokens directly to generated tokens in many
        # OpenAI-compatible servers. We use this for very long prompts when the
        # chat endpoint may compute an invalid default_max_tokens internally.
        # Some TensorRT-LLM versions ignore streaming for /v1/completions and
        # return one normal JSON object instead of SSE. The parser below handles
        # both cases.
        completion_stream = _env_flag("COMPLETION_STREAM", "0")
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": completion_stream,
        }
    else:
        completion_stream = True
        # TensorRT-LLM 1.1.0's OpenAI chat endpoint rejects some newer
        # OpenAI request fields. In particular, max_completion_tokens can cause
        # an immediate HTTP 400 on otherwise valid 1k/8k requests. Therefore
        # the default chat payload uses only max_tokens. If a newer server needs
        # max_completion_tokens, enable it explicitly with
        # INCLUDE_MAX_COMPLETION_TOKENS=1.
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if os.environ.get("INCLUDE_MAX_COMPLETION_TOKENS", "0") == "1":
            payload["max_completion_tokens"] = max_tokens

    start = time.perf_counter()
    first_token_time = None
    end = None
    output_text = ""
    completion_tokens = None
    prompt_tokens_reported = None
    error = None
    status_code = None

    try:
        connect_timeout_s = float(os.environ.get("REQUEST_CONNECT_TIMEOUT_S", "10"))
        read_timeout_s = float(os.environ.get("REQUEST_READ_TIMEOUT_S", os.environ.get("FIRST_TOKEN_TIMEOUT_S", str(timeout_s))))
        _debug_print(
            f"[request {request_id}] Sending {api_mode} request to server "
            f"(connect_timeout={connect_timeout_s}s, read_timeout={read_timeout_s}s, "
            f"stream={payload.get('stream')})..."
        )

        if not payload.get("stream"):
            # Non-streaming completion path: TTFT is not observable, but total latency
            # and output tokens/TPS are recorded reliably.
            r = requests.post(url, json=payload, timeout=(connect_timeout_s, read_timeout_s))
            status_code = r.status_code
            _debug_print(f"[request {request_id}] HTTP status={status_code}; parsing full JSON response...")
            if status_code >= 400:
                body = r.text[:2000]
                error = f"HTTPError({status_code}): {body}"
                end = time.perf_counter()
            else:
                obj = r.json()
                chunk_text, c_toks, p_toks = _extract_text_and_usage_from_response(obj, api_mode)
                output_text += chunk_text
                completion_tokens = c_toks if c_toks is not None else completion_tokens
                prompt_tokens_reported = p_toks if p_toks is not None else prompt_tokens_reported
                end = time.perf_counter()
        else:
            non_sse_lines = []
            saw_sse = False
            with requests.post(url, json=payload, stream=True, timeout=(connect_timeout_s, read_timeout_s)) as r:
                status_code = r.status_code
                _debug_print(f"[request {request_id}] HTTP status={status_code}; waiting for stream tokens...")
                if status_code >= 400:
                    try:
                        body = r.text[:2000]
                    except Exception as body_exc:
                        body = f"<could not read response body: {body_exc!r}>"
                    _debug_print(f"[request {request_id}] HTTP error body: {body}")
                    error = f"HTTPError({status_code}): {body}"
                    end = time.perf_counter()
                    return {
                        "request_id": request_id,
                        "success": False,
                        "ttft_ms": None,
                        "tps": None,
                        "output_tokens": 0,
                        "latency_s": end - start,
                        "error": error,
                        "target_prompt_tokens": target_prompt_tokens,
                        "prompt_tokens_est": prompt_tokens_est,
                        "prompt_tokens_reported": prompt_tokens_reported,
                    }
                r.raise_for_status()

                for raw_line in r.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    line = raw_line.strip()
                    if not line.startswith("data: "):
                        # Some TensorRT-LLM completion responses are normal JSON even
                        # when stream=True is requested. Keep them and parse after loop.
                        non_sse_lines.append(line)
                        continue

                    saw_sse = True
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue

                    chunk_text, c_toks, p_toks = _extract_text_and_usage_from_response(obj, api_mode)
                    if c_toks is not None:
                        completion_tokens = c_toks
                    if p_toks is not None:
                        prompt_tokens_reported = p_toks
                    if chunk_text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            _debug_print(
                                f"[request {request_id}] First token after "
                                f"{first_token_time - start:.2f}s"
                            )
                        output_text += chunk_text

                end = time.perf_counter()

            # Normal JSON fallback for /v1/completions when no SSE data arrived.
            # Some servers send one full JSON object; others send multiple JSON
            # lines without SSE prefixes. Try both patterns.
            if not output_text and non_sse_lines:
                raw = "\n".join(non_sse_lines).strip()
                parsed_any = False
                candidates = [raw] + [x.strip() for x in non_sse_lines if x.strip()]
                for candidate in candidates:
                    if not candidate:
                        continue
                    try:
                        obj = json.loads(candidate)
                    except Exception:
                        continue
                    parsed_any = True
                    chunk_text, c_toks, p_toks = _extract_text_and_usage_from_response(obj, api_mode)
                    output_text += chunk_text
                    if c_toks is not None:
                        completion_tokens = c_toks
                    if p_toks is not None:
                        prompt_tokens_reported = p_toks
                if output_text and first_token_time is None:
                    # This is a full-response fallback, so true TTFT is unavailable.
                    # We use full response latency as an upper bound rather than leaving
                    # TTFT blank, and the report should label completion-mode TTFT this way.
                    first_token_time = end
                    _debug_print(
                        f"[request {request_id}] Parsed non-SSE JSON response; "
                        f"TTFT is recorded as full-response upper bound."
                    )
                elif not parsed_any:
                    preview = raw[:1000].replace("\n", "\\n")
                    _debug_print(f"[request {request_id}] Could not parse non-SSE response as JSON. Preview: {preview}")

    except Exception as e:
        end = time.perf_counter()
        error = repr(e)
        if first_token_time is None and ("Read timed out" in error or "ReadTimeout" in error):
            error = "first_token_timeout_or_server_stall: " + error
        _debug_print(f"[request {request_id}] Request failed after {end - start:.2f}s: {error}")

    ttft_ms = None
    if first_token_time is not None:
        ttft_ms = (first_token_time - start) * 1000.0

    total_time_s = max((end or time.perf_counter()) - start, 1e-9)

    tokenizer_path = os.environ.get("TOKENIZER_PATH") or os.environ.get("PLAN_MODEL") or os.environ.get("MODEL_PATH")
    if completion_tokens is None:
        completion_tokens = _count_output_tokens_fallback(output_text, tokenizer_path)

    # A benchmark request that returns no generated tokens is not a useful pass.
    # This catches parser mismatches and silent empty completions.
    if error is None and max_tokens > 0 and completion_tokens == 0:
        error = f"no_output_tokens_returned_or_parsed_status_{status_code}_api_{api_mode}"
        _debug_print(
            f"[request {request_id}] No output tokens parsed. "
            f"api_mode={api_mode}, status_code={status_code}, output_chars={len(output_text)}, "
            f"prompt_tokens_est={prompt_tokens_est}"
        )

    tps = completion_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "request_id": request_id,
        "success": error is None,
        "status_code": status_code,
        "error": error or "",
        "ttft_ms": ttft_ms,
        "total_time_s": total_time_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens_est": prompt_tokens_est,
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_tokens_reported": prompt_tokens_reported,
        "tps": tps,
        "output_chars": len(output_text),
    }

def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(q * (len(values) - 1))
    return values[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--framework", default="tensorrt-llm")
    parser.add_argument("--quantization", default="smoke-test")
    parser.add_argument("--decode-mode", default="baseline", help="baseline, draft_target, eagle3, etc.")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--output", default="results/smoke_results.csv")
    parser.add_argument("--api-mode", default=os.environ.get("OPENAI_API_MODE", "chat"), choices=["chat", "completion", "completions"], help="Use /v1/chat/completions or /v1/completions")
    parser.add_argument("--scenario-name", default=os.environ.get("SCENARIO_NAME", "unspecified"))
    parser.add_argument("--workload-type", default=os.environ.get("WORKLOAD_TYPE", "unspecified"))
    parser.add_argument("--prompt-profile", default=os.environ.get("PROMPT_PROFILE", "synthetic_code_context"))
    parser.add_argument("--duration-s", type=float, default=float(os.environ.get("DURATION_S", "0")))
    args = parser.parse_args()
    os.environ["PROMPT_PROFILE"] = args.prompt_profile

    api_mode = "completion" if args.api_mode == "completions" else args.api_mode
    endpoint = "completions" if api_mode == "completion" else "chat/completions"
    url = f"http://{args.host}:{args.port}/v1/{endpoint}"

    idle_gpu = gpu_snapshot()
    vram_idle_gb = sum_gpu_mem_gb(idle_gpu)

    results = []
    start_wall = time.perf_counter()

    print(
        f"Starting benchmark requests: context={args.context_len}, "
        f"concurrency={args.concurrency}, num_requests={args.num_requests}, "
        f"max_tokens={args.max_tokens}, timeout_s={args.timeout_s}, api_mode={api_mode}, "
        f"scenario={args.scenario_name}, workload={args.workload_type}, prompt_profile={args.prompt_profile}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(
                run_one_request,
                url,
                args.model,
                args.context_len,
                args.max_tokens,
                i,
                args.timeout_s,
                api_mode,
            )
            for i in range(args.num_requests)
        ]

        completed = 0
        pending = set(futures)
        heartbeat_s = float(os.environ.get("BENCHMARK_HEARTBEAT_S", "60"))
        last_heartbeat = time.perf_counter()

        while pending:
            done, pending = wait(pending, timeout=heartbeat_s, return_when=FIRST_COMPLETED)

            if not done:
                now = time.perf_counter()
                snap = gpu_snapshot()
                gpu_util = mean_gpu_util(snap)
                vram_now = sum_gpu_mem_gb(snap)
                print(
                    f"[heartbeat] {len(pending)}/{args.num_requests} requests still running; "
                    f"elapsed={now - start_wall:.1f}s; "
                    f"vram_used_total_gb={vram_now:.2f}; gpu_util_mean={gpu_util}",
                    flush=True,
                )
                stall_reason = detect_server_log_stall(args.context_len + args.max_tokens)
                if stall_reason:
                    print(f"[heartbeat] Detected server-side long-context stall: {stall_reason}", flush=True)
                    # Exit with a special code so run_benchmark_grid.sh records a clean failure row.
                    os._exit(88)
                last_heartbeat = now
                continue

            for fut in done:
                result = fut.result()
                results.append(result)
                completed += 1
                status = "ok" if result.get("success") else "fail"
                print(
                    f"Completed request {completed}/{args.num_requests} "
                    f"(id={result.get('request_id')}, status={status}, "
                    f"ttft_ms={result.get('ttft_ms')}, "
                    f"total_time_s={result.get('total_time_s'):.2f}, "
                    f"prompt_tokens={result.get('prompt_tokens_reported') or result.get('prompt_tokens_est')})",
                    flush=True,
                )

    end_wall = time.perf_counter()

    load_gpu = gpu_snapshot()
    vram_load_gb = sum_gpu_mem_gb(load_gpu)

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    ttfts = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]
    tps_vals = [r["tps"] for r in successes if r["tps"] is not None]
    total_output_tokens = sum(r["completion_tokens"] for r in successes)
    total_time_s = max(end_wall - start_wall, 1e-9)
    aggregate_tps = total_output_tokens / total_time_s

    prompt_est_values = [r.get("prompt_tokens_est") for r in results if r.get("prompt_tokens_est") not in (None, -1)]
    prompt_reported_values = [r.get("prompt_tokens_reported") for r in results if r.get("prompt_tokens_reported") is not None]
    target_prompt_values = [r.get("target_prompt_tokens") for r in results if r.get("target_prompt_tokens") is not None]

    row = {
        "scenario_name": args.scenario_name,
        "workload_type": args.workload_type,
        "prompt_profile": args.prompt_profile,
        "api_mode": api_mode,
        "duration_s_requested": args.duration_s,
        "framework": args.framework,
        "model": args.model,
        "quantization": args.quantization,
        "decode_mode": args.decode_mode,
        "gpu_type": "; ".join(sorted(set(g.get("gpu_name", "unknown") for g in load_gpu))),
        "num_gpus": len([g for g in load_gpu if "gpu_index" in g]),
        "context_len": args.context_len,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_tokens,
        "target_prompt_tokens": max(target_prompt_values) if target_prompt_values else "",
        "prompt_tokens_est": max(prompt_est_values) if prompt_est_values else "",
        "prompt_tokens_reported": max(prompt_reported_values) if prompt_reported_values else "",
        "num_requests": args.num_requests,
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else "",
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else "",
        "ttft_p99_ms": percentile(ttfts, 0.99) if ttfts else "",
        "tps_mean": statistics.mean(tps_vals) if tps_vals else "",
        "tps_p50": statistics.median(tps_vals) if tps_vals else "",
        "tps_p99": percentile(tps_vals, 0.99) if tps_vals else "",
        "aggregate_tps": aggregate_tps,
        "total_output_tokens": total_output_tokens,
        "total_time_s": total_time_s,
        "vram_idle_gb": vram_idle_gb,
        "vram_load_gb": vram_load_gb,
        "kv_cache_growth_gb": max(0.0, vram_load_gb - vram_idle_gb),
        "gpu_util_mean_after": mean_gpu_util(load_gpu),
        "runtime_stability": "pass" if len(failures) == 0 else "fail",
        "error_count": len(failures),
        "error_messages": " | ".join(sorted(set(f["error"] for f in failures if f["error"]))),
    }

    fieldnames = list(row.keys())
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

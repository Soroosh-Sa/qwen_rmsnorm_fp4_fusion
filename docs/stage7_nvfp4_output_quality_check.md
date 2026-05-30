# Stage 7: NVFP4 output quality and equivalence sanity check

The Stage-6 benchmark proves that the engines run and reports throughput. It does **not** prove that the generated text is equivalent or meaningful.

Stage 7 compares outputs from the same deterministic prompt set across:

- **A:** normal/original NVFP4 TensorRT-LLM engine
- **B:** folded-weight NVFP4 base engine, no plugin
- **C:** folded-weight NVFP4 plugin engine

The goal is to check:

1. outputs are non-empty,
2. outputs do not look obviously corrupted/repetitive,
3. outputs satisfy simple expected-keyword checks,
4. C is broadly similar to A and B for the same prompts.

This is a sanity check, not a full accuracy benchmark. For a paper-quality result, follow this with task-level evaluations such as MMLU-style QA, HumanEval/code tasks, perplexity on held-out text, or exact-match tests for deterministic prompts.

## Run

```bash
cd ~/workspace/qwen_rmsnorm_fp4_fusion

export MODEL_PROFILE=qwen25_05b
export TP_SIZE=2
export OPENAI_API_MODE=completion
export COMPLETION_USE_CHAT_TEMPLATE=1
export QUALITY_MAX_NEW_TOKENS=128

bash scripts/run_nvfp4_three_way_quality_check.sh
```

## Outputs

```text
results/<MODEL_TAG>_normal_nvfp4_quality_outputs.jsonl
results/<MODEL_TAG>_folded_nvfp4_base_quality_outputs.jsonl
results/<MODEL_TAG>_folded_nvfp4_plugin_quality_outputs.jsonl
results/<MODEL_TAG>_nvfp4_three_way_quality_summary.csv
results/<MODEL_TAG>_nvfp4_three_way_quality_report.md
```

## How to read the report

Important fields:

- `plugin_expected_pass`: plugin output contains the required keywords for that prompt.
- `plugin_garbage_flags`: should be empty.
- `plugin_vs_normal_char_similarity`: rough string similarity between C and A.
- `plugin_vs_folded_base_char_similarity`: rough string similarity between C and B.
- `plugin_close_to_normal`: boolean heuristic based on character similarity or token overlap.
- `plugin_close_to_folded_base`: boolean heuristic based on character similarity or token overlap.

Because LLM outputs can differ even under greedy decoding after quantization, exact match is not required. The useful signs are: no garbage flags, expected-keyword pass, and broadly similar outputs.

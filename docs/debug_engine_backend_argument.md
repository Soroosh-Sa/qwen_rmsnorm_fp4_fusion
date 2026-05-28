# Engine serving backend argument note

For a built TensorRT-LLM engine directory, the safest command form in the NGC TensorRT-LLM containers is:

```bash
trtllm-serve serve /path/to/engine_dir \
  --tokenizer /path/to/tokenizer_dir \
  --host 0.0.0.0 \
  --port 8000
```

Do not pass `--backend pytorch`; that forces the PyTorch checkpoint loader.

Do not pass `--backend tensorrt` by default. In some TensorRT-LLM builds this still routes through the LLM API checkpoint-loading/build path and can fail with:

```text
assert os.path.isfile(weights_path)
AssertionError
```

The repo's `scripts/serve_engine_target.sh` now avoids `--backend` by default for engine directories. To override for debugging:

```bash
TRTLLM_ENGINE_BACKEND=pytorch bash scripts/serve_engine_target.sh
```

For normal engine serving, leave `TRTLLM_ENGINE_BACKEND` unset.

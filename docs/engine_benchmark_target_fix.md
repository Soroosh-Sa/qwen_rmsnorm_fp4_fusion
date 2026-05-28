# Engine benchmark target fix

If the server reaches `Server health check passed` and then the script prints:

```text
Unknown TARGET=selected_original_nvfp4_engine
```

then the TensorRT-LLM engine serving step worked, but the benchmark wrapper did
not know how to label/run the new engine target.

This is fixed by adding these target names to `scripts/run_quantized_target_benchmark.sh`:

- `selected_original_nvfp4_engine`
- `selected_folded_nvfp4_engine`

These targets use the already running engine server. Their `PLAN_MODEL` is only
used by the Python benchmark client for tokenizer metadata and prompt sizing.
The actual served model is the `ENGINE_DIR` started by `scripts/serve_engine_target.sh`.

A successful run should show:

```text
Server health check passed.
Models:
...
Running benchmark grid...
```

instead of stopping at `Unknown TARGET`.

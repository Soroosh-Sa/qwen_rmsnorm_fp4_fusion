# Next Steps After Smoke Test

1. Run `scripts/01_verify_folding.sh` on a tiny Qwen model.
2. Check `metrics/folding_layer_checks.csv`.
3. Run `scripts/02_fold_checkpoint.sh`.
4. Run `scripts/03_verify_model_logits.sh`.
5. Confirm last-token top-1 mostly matches and errors are small.
6. Add the exact ModelOpt/TensorRT-LLM FP4 command to `scripts/04_quantize_modelopt_placeholder.sh`.
7. Quantize the folded checkpoint.
8. Compare original quantized vs folded quantized accuracy/perplexity.
9. Only then scale the approach to larger Qwen.

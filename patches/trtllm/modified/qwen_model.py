# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import ctypes
import os
from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from tqdm import tqdm

from ..._utils import pad_vocab_size
from ...functional import ACT2FN, LayerNormType, Tensor, cast, mean, sqrt, split, recv, send, default_trtnet, _create_tensor
from ...layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, GatedMLP, RmsNorm, SharedMoE)
from ...layers.moe import MOEWeightWrapper
from ...logger import logger
from ...lora_helper import (LoraConfig,
                            get_default_trtllm_modules_to_hf_modules, use_lora)
from ...mapping import Mapping
from ...module import Module
from ...quantization import QuantAlgo
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig)
from .config import QWenConfig
from .convert import (load_hf_qwen, load_weights_from_hf_gptq_model,
                      load_weights_from_hf_model)


_QWEN_RMS_SCALE_SWIGLU_PLUGIN_LOADED = False


def _load_qwen_rms_scale_swiglu_plugin():
    """Load the external TensorRT plugin .so once during engine build/runtime."""
    global _QWEN_RMS_SCALE_SWIGLU_PLUGIN_LOADED

    if _QWEN_RMS_SCALE_SWIGLU_PLUGIN_LOADED:
        return

    so_path = os.environ.get("QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO", "")
    if not so_path:
        raise RuntimeError(
            "QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO is not set, but "
            "TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN=1 was requested."
        )

    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    _QWEN_RMS_SCALE_SWIGLU_PLUGIN_LOADED = True


def _create_qwen_rms_scale_swiglu_plugin(plugin_name, layer_name, fields, input_tensors):
    _load_qwen_rms_scale_swiglu_plugin()

    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator(plugin_name, "1", "")
    if creator is None:
        raise RuntimeError(
            f"Could not find TensorRT plugin creator {plugin_name} v1. "
            "Check QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO and plugin registration."
        )

    plugin = creator.create_plugin(
        layer_name,
        trt.PluginFieldCollection(fields),
    )

    layer = default_trtnet().add_plugin_v2(
        [tensor.trt_tensor for tensor in input_tensors],
        plugin,
    )

    return _create_tensor(layer.get_output(0), layer)


def _qwen_rms_scale_swiglu_fields(hidden_size, inter_size, eps):
    return [
        trt.PluginField(
            "hidden_size",
            np.array([hidden_size], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "inter_size",
            np.array([inter_size], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "eps",
            np.array([eps], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
    ]


def qwen_rms_scale_swiglu_fused_plugin(hidden_states, raw_gate_up, hidden_size, inter_size, eps):
    """
    FusedGatedMLP plugin. Inputs:
      hidden_states, raw_gate_up = [raw_gate | raw_inter]

    Computes:
      rstd = rsqrt(mean(hidden_states^2) + eps)
      out = silu(raw_inter * rstd) * (raw_gate * rstd)
    """
    fields = _qwen_rms_scale_swiglu_fields(hidden_size, inter_size, eps)
    return _create_qwen_rms_scale_swiglu_plugin(
        "QwenRmsScaleSwiglu",
        "qwen_rms_scale_swiglu_fused",
        fields,
        [hidden_states, raw_gate_up],
    )


def qwen_rms_scale_swiglu_gated_plugin(hidden_states, raw_gate, raw_inter, hidden_size, inter_size, eps):
    """
    Plain GatedMLP plugin. Inputs:
      hidden_states, raw_gate, raw_inter

    This avoids a TensorRT concat before the plugin and is the preferred path
    when TensorRT-LLM keeps fc/gate as separate projections for NVFP4.
    """
    fields = _qwen_rms_scale_swiglu_fields(hidden_size, inter_size, eps)
    return _create_qwen_rms_scale_swiglu_plugin(
        "QwenRmsScaleSwigluGated",
        "qwen_rms_scale_swiglu_gated",
        fields,
        [hidden_states, raw_gate, raw_inter],
    )


# Backward-compatible alias used by older local patches.
def qwen_rms_scale_swiglu_plugin(hidden_states, raw_gate_up, hidden_size, inter_size, eps):
    return qwen_rms_scale_swiglu_fused_plugin(
        hidden_states, raw_gate_up, hidden_size, inter_size, eps
    )


class QWenDecoderLayer(Module):

    def __init__(self, config: QWenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        dtype = config.dtype
        self.tp_group = config.mapping.tp_group
        self.tp_size = config.mapping.tp_size

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        # Qwen3: Enable qk_layernorm for Q/K normalization (similar to Gemma3)
        qk_layernorm = config.qwen_type in ('qwen3', 'qwen3_moe')

        self.attention = Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_seqlen_for_logn_scaling=config.seq_length,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_rank=config.mapping.tp_rank,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
            cp_rank=config.mapping.cp_rank,
            cp_size=config.mapping.cp_size,
            cp_group=config.mapping.cp_group,
            quant_mode=config.quant_mode,
            use_logn_scaling=config.use_logn_attn,
            dense_bias=False,
            # Qwen3: Add Q/K layer normalization
            qk_layernorm=qk_layernorm,
            layernorm_type=LayerNormType.RmsNorm
            if qk_layernorm else LayerNormType.LayerNorm)

        if config.moe.has_moe():
            mlp_kwargs = {'moe_config': config.moe, 'mapping': config.mapping}
            if config.qwen_type == 'qwen2_moe':
                # Qwen2 MoE uses SharedMoE with shared expert
                ClsMLP = SharedMoE
                mlp_kwargs['use_shared_gate'] = True
                mlp_kwargs['use_side_stream'] = True
                mlp_kwargs['moe_config'].shared_expert_intermediate_size = \
                    config.moe_shared_expert_intermediate_size
            elif config.qwen_type == 'qwen3_moe':
                # Qwen3 MoE uses standard MOE without shared expert
                ClsMLP = MOE
            else:
                ClsMLP = MOE
        else:
            ClsMLP = GatedMLP
            mlp_kwargs = {}

        # Qwen's real inter_size depends on qwen_type
        if self.config.qwen_type == 'qwen':
            intermediate_size = config.intermediate_size // 2
        elif self.config.qwen_type in ('qwen2_moe', 'qwen3_moe'):
            intermediate_size = config.moe_intermediate_size
        else:
            intermediate_size = config.intermediate_size

        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=intermediate_size,
                          hidden_act=config.hidden_act,
                          dtype=dtype,
                          bias=config.mlp_bias,
                          tp_group=self.tp_group,
                          tp_size=self.tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=dtype)

    def _use_folded_rmsnorm_mlp_fusion(self, lora_layer_params=None):
        """
        Experimental Runara-style folded RMSNorm + MLP fusion.

        This path is only correct when the checkpoint has already folded
        post_attention_layernorm.weight into mlp.fc and mlp.gate, and the
        post_attention_layernorm.weight itself has been reset to 1.

        Enable with:
            TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION=1

        By default this is BF16/FP16-only. For NVFP4 experiments later:
            TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION_ALLOW_QUANTIZED=1
        """
        if os.environ.get("TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION", "0") != "1":
            return False

        if lora_layer_params is not None:
            raise NotImplementedError(
                "TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION does not support LoRA yet."
            )

        # Only handle dense Qwen/Qwen2 MLP variants first.
        # Supported for now:
        #   1. GatedMLP      : fc + gate + proj
        #   2. FusedGatedMLP : fused_fc + proj
        #
        # MoE / SharedMoE will be handled later.
        has_gated_mlp = all(
            hasattr(self.mlp, name) for name in ("fc", "gate", "proj")
        )

        has_fused_gated_mlp = all(
            hasattr(self.mlp, name) for name in ("fused_fc", "proj")
        )

        if not has_gated_mlp and not has_fused_gated_mlp:
            mlp_attrs = sorted([
                name for name in dir(self.mlp)
                if not name.startswith("_")
            ])
            raise NotImplementedError(
                "TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION currently supports only dense "
                "GatedMLP or FusedGatedMLP. "
                f"Actual mlp class: {self.mlp.__class__.__name__}. "
                f"Available attrs: {mlp_attrs[:80]}"
            )


        if not hasattr(self.mlp, "hidden_act") and not hasattr(self.config, "hidden_act"):
            raise NotImplementedError(
                "TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION could not find hidden_act "
                "on self.mlp or self.config."
            )

        if getattr(self.mlp, "inner_layernorm", None) is not None:
            raise NotImplementedError(
                "TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION does not support inner_layernorm yet."
            )

        # Bias is important mathematically:
        # correct: (x @ W.T) / rms + b
        # wrong:   (x @ W.T + b) / rms
        # The current graph-level patch calls self.mlp.fc(x) and self.mlp.gate(x),
        # so it is only exact when those projections are bias-free.
        if getattr(self.config, "mlp_bias", False) or getattr(self.mlp, "bias", False):
            raise NotImplementedError(
                "TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION currently requires bias-free MLP projections."
            )

        quantization = getattr(self.config, "quantization", None)
        quant_algo = getattr(quantization, "quant_algo", None)
        allow_quantized = os.environ.get(
            "TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION_ALLOW_QUANTIZED", "0"
        ) == "1"

        if quant_algo is not None and not allow_quantized:
            raise NotImplementedError(
                "TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION is currently enabled only for BF16/FP16. "
                "For NVFP4 experiments, set "
                "TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION_ALLOW_QUANTIZED=1."
            )

        return True

    def _folded_rmsnorm_mlp_fusion_forward(self, hidden_states):
        """
        Folded-weight equivalent of:
            hidden_states = post_layernorm(hidden_states)
            hidden_states = mlp(hidden_states)

        For plugin mode, the fused_fc GEMM runs on the unnormalized residual
        hidden_states using weights that already contain RMSNorm gamma. The
        plugin then applies only the RMS denominator scaling and SwiGLU.
        """
        hidden_act = getattr(self.mlp, "hidden_act",
                             getattr(self.config, "hidden_act", None))

        use_plugin = os.environ.get("TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN", "0") == "1"

        if hasattr(self.mlp, "fused_fc"):
            # FusedGatedMLP path:
            # fused_fc output shape is [..., 2 * ffn_hidden_size / tp_size].
            raw_gate_up = self.mlp.fused_fc(hidden_states)
            split_size = self.mlp.ffn_hidden_size // self.mlp.tp_size

            if use_plugin:
                intermediate = qwen_rms_scale_swiglu_fused_plugin(
                    hidden_states,
                    raw_gate_up,
                    self.config.hidden_size,
                    split_size,
                    self.post_layernorm.eps,
                )
            else:
                x_fp32 = cast(hidden_states, "float32")
                rms = sqrt(
                    mean(x_fp32 * x_fp32, dim=-1, keepdim=True) +
                    self.post_layernorm.eps
                )
                rms = cast(rms, self.config.dtype)

                raw_gate_up = raw_gate_up / rms
                gate, inter = split(raw_gate_up, split_size, dim=-1)

                inter = ACT2FN[hidden_act](inter)
                intermediate = inter * gate
        else:
            # Plain GatedMLP path. This is common in NVFP4 TensorRT-LLM builds.
            # We use a separate 3-input plugin instead of concat([gate, inter])
            # to avoid materializing another tensor.
            split_size = self.mlp.ffn_hidden_size // self.mlp.tp_size

            raw_inter = self.mlp.fc(hidden_states)
            raw_gate = self.mlp.gate(hidden_states)

            if use_plugin:
                intermediate = qwen_rms_scale_swiglu_gated_plugin(
                    hidden_states,
                    raw_gate,
                    raw_inter,
                    self.config.hidden_size,
                    split_size,
                    self.post_layernorm.eps,
                )
            else:
                # Plain graph-level folded fallback.
                x_fp32 = cast(hidden_states, "float32")
                rms = sqrt(
                    mean(x_fp32 * x_fp32, dim=-1, keepdim=True) +
                    self.post_layernorm.eps
                )
                rms = cast(rms, self.config.dtype)

                inter = raw_inter / rms
                gate = raw_gate / rms

                inter = ACT2FN[hidden_act](inter)
                intermediate = inter * gate

        if getattr(self.mlp, "inner_layernorm", None) is not None:
            intermediate = self.mlp.inner_layernorm(intermediate)

        output = self.mlp.proj(intermediate)
        return output

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        spec_decoding_params=None,
        kv_cache_params=None,
        attention_params=None,
        lora_layer_params=None,
        mrope_params=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
            mrope_params=mrope_params,
        )
        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states

        if self._use_folded_rmsnorm_mlp_fusion(lora_layer_params):
            hidden_states = self._folded_rmsnorm_mlp_fusion_forward(hidden_states)
        else:
            hidden_states = self.post_layernorm(hidden_states)

            hidden_states = self.mlp(hidden_states,
                                     lora_layer_params=lora_layer_params)

        hidden_states = residual + hidden_states
        
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class QWenModel(Module):

    def __init__(self, config: QWenConfig) -> None:
        super().__init__()
        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(QWenDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                spec_decoding_params=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                mrope_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_params=lora_params,
            mrope_params=mrope_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class QWenForCausalLM(DecoderModelForCausalLM):
    config_class = QWenConfig

    def __init__(self, config: QWenConfig):
        transformer = QWenModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        if config.mapping.is_last_pp_rank():
            if config.architecture == 'Qwen2ForSequenceClassification':
                lm_head = ColumnLinear(config.hidden_size,
                                       config.num_labels,
                                       bias=False,
                                       dtype=config.dtype,
                                       tp_group=config.mapping.tp_group,
                                       tp_size=config.mapping.tp_size,
                                       gather_output=True)
            else:
                lm_head = ColumnLinear(config.hidden_size,
                                       vocab_size_padded,
                                       bias=False,
                                       dtype=config.dtype,
                                       tp_group=config.mapping.tp_group,
                                       tp_size=config.mapping.tp_size,
                                       gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        if config.qwen_type == 'qwen':
            self.trtllm_modules_to_hf_modules = {
                "attn_qkv": "c_attn",
                "attn_dense": "attn.c_proj",
                "mlp_h_to_4h": "w2",
                "mlp_4h_to_h": "mlp.c_proj",
                "mlp_gate": "w1",
            }
        elif config.qwen_type in ('qwen2_moe', 'qwen3_moe'):
            self.trtllm_modules_to_hf_modules = copy.copy(
                get_default_trtllm_modules_to_hf_modules())
            # Common MoE expert mappings for both Qwen2 and Qwen3 MoE
            self.trtllm_modules_to_hf_modules.update({
                "moe_h_to_4h":
                "mlp.experts.gate_proj",
                "moe_4h_to_h":
                "mlp.experts.down_proj",
                "moe_gate":
                "mlp.experts.up_proj",
            })
            # Qwen2 MoE additionally has shared expert
            if config.qwen_type == 'qwen2_moe':
                self.trtllm_modules_to_hf_modules.update({
                    "mlp_h_to_4h":
                    "mlp.shared_expert.gate_proj",
                    "mlp_4h_to_h":
                    "mlp.shared_expert.down_proj",
                    "mlp_gate":
                    "mlp.shared_expert.up_proj",
                    "mlp_router":
                    "mlp.shared_expert_gate",
                })
        else:
            self.trtllm_modules_to_hf_modules = None
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        ''' Create a QWenForCausalLM object from give parameters
        '''
        import transformers

        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
        use_autoawq = kwargs.pop('use_autoawq', False)

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = QWenConfig.from_hugging_face(hf_config_or_dir,
                                              dtype=dtype,
                                              mapping=mapping,
                                              quant_config=quant_config,
                                              **kwargs)

        if os.environ.get("TRTLLM_DISABLE_UNIFIED_CONVERTER") is None:
            arg_dict = {"use_autoawq": True} if use_autoawq else {}
            custom_dict = {}

            if config.qwen_type == "qwen":
                custom_dict = {
                    "transformer": "transformer",
                    "vocab_embedding": "wte",
                    "ln_f": "ln_f",
                    "layers": "h",
                    "attention": "attn",
                    "qkv": "c_attn",
                    "dense": "c_proj",
                    "gate": "w1",
                    "proj": "c_proj",
                    "fc": "w2",
                    "input_layernorm": "ln_1",
                    "post_layernorm": "ln_2",
                }
            elif config.qwen_type == "qwen2_moe":
                custom_dict = {
                    "mlp.shared_expert": "mlp.shared_expert",
                    "mlp.shared_expert_gate": "mlp.shared_expert_gate",
                    "fc": ["up_proj", "gate_proj"],
                }
            elif config.qwen_type == "qwen3_moe":
                custom_dict = {
                    "fc": ["up_proj", "gate_proj"],
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
            elif config.qwen_type in {"qwen2", "qwen2_vl"
                                      } and config.tie_word_embeddings:
                custom_dict = {"lm_head": "model.embed_tokens"}
            elif config.architecture == "Qwen2ForSequenceClassification":
                custom_dict = {
                    "lm_head": "score",
                }
            elif config.qwen_type == "qwen2_llava_onevision":
                custom_dict = {
                    "transformer": "language_model.model",
                    "lm_head": "language_model.lm_head",
                }
            elif config.qwen_type == "qwen2_audio":
                custom_dict = {
                    "transformer": "language_model.model",
                    "lm_head": "language_model.lm_head",
                }
            elif config.qwen_type == "qwen3":
                custom_dict = {
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
            loader = ModelWeightsLoader(hf_model_dir, custom_dict)
            model = cls(config)
            if config.qwen_type == "qwen" and model.config.mapping.has_tp():

                def reshape_qkv(weights):
                    if weights is None:
                        return weights
                    mapping = model.config.mapping
                    unsqueeze = False
                    if isinstance(weights, torch.Tensor):
                        unsqueeze = True
                        weights = [weights]

                    for idx, w in enumerate(weights):
                        if quant_config.quant_algo == QuantAlgo.W4A16_GPTQ:
                            w = w.reshape(-1, 3, w.shape[-1] // 3)
                            w = w.chunk(mapping.tp_size, 2)[mapping.tp_rank]
                            if w.shape[0] == 1:
                                weights[idx] = w.reshape(-1)
                            else:
                                weights[idx] = w.reshape(w.shape[0], -1)
                        else:
                            w = w.reshape(3, w.shape[0] // 3, -1)
                            w = w.chunk(mapping.tp_size, 1)[mapping.tp_rank]
                            if w.shape[-1] == 1:
                                weights[idx] = w.reshape(-1)
                            else:
                                weights[idx] = w.reshape(-1, w.shape[-1])
                    if unsqueeze:
                        return weights[0]
                    else:
                        return weights

                loader.update_key_mapping(model)
                tllm_weights = {}
                for tllm_key, _ in tqdm(model.named_parameters()):
                    if "qkv" in tllm_key:
                        tllm_weights.update(
                            loader.load(tllm_key,
                                        reshape_qkv,
                                        skip_tp=True,
                                        custom_postprocess_kwargs=arg_dict))
                    else:
                        tllm_weights.update(
                            loader.load(tllm_key,
                                        custom_postprocess_kwargs=arg_dict))
                loader.fill(tllm_weights)
            elif config.qwen_type in ("qwen2_moe", "qwen3_moe"):
                for tllm_key, _ in model.named_parameters():
                    sub_module = model
                    for attr in tllm_key.split(".")[:-1]:
                        sub_module = getattr(sub_module, attr)
                    if "router" in tllm_key or isinstance(
                            sub_module, MOEWeightWrapper):
                        sub_module_dic = sub_module.tllm_to_externel_key_dict
                        sub_module_dic["mlp"] = "mlp"
                        if "fc" in sub_module_dic.keys():
                            sub_module_dic["fc"] = [
                                hf_keyword.replace("w1", "gate_proj")
                                for hf_keyword in sub_module_dic["fc"]
                            ]
                            sub_module_dic["fc"] = [
                                hf_keyword.replace("w3", "up_proj")
                                for hf_keyword in sub_module_dic["fc"]
                            ]
                        if "proj" in sub_module_dic.keys():
                            sub_module_dic["proj"] = [
                                hf_keyword.replace("w2", "down_proj")
                                for hf_keyword in sub_module_dic["proj"]
                            ]
                        sub_module.tllm_to_externel_key_dict = sub_module_dic

                def concat_gate_up_proj(weights):
                    return torch.cat(weights, dim=-2)

                loader.update_key_mapping(model)
                tllm_weights = {}
                for tllm_key, _ in tqdm(model.named_parameters()):
                    if tllm_key.endswith("shared_expert.fc.weight"):
                        tllm_weights.update(
                            loader.load(tllm_key,
                                        concat_gate_up_proj,
                                        custom_postprocess_kwargs=arg_dict))
                    else:
                        tllm_weights.update(
                            loader.load(tllm_key,
                                        custom_postprocess_kwargs=arg_dict))
                loader.fill(tllm_weights)
            else:
                # For Qwen1 w/o TP, Qwen1.5 and Qwen2 w/o MoE
                loader.generate_tllm_weights(model, arg_dict)
        else:
            if not use_preloading:
                hf_model = load_hf_qwen(hf_model_dir, load_model_on_cpu)

            logger.debug(f"HuggingFace model: {hf_model}")

            model = QWenForCausalLM(config)
            logger.debug(f"TensorRT LLM model: {model}")

            if quant_config.quant_algo == QuantAlgo.W4A16_GPTQ:
                weights = load_weights_from_hf_gptq_model(hf_model, config)
            else:
                weights = load_weights_from_hf_model(hf_model, config)
            model.load(weights)
        return model

    def default_plugin_config(self, **kwargs):
        plugin_config = super().default_plugin_config(**kwargs)
        if self.quant_mode.is_int4_weight_only_per_group():
            plugin_config.weight_only_groupwise_quant_matmul_plugin = 'auto'
        return plugin_config

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = 'auto',
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        calib_dataset='cnn_dailymail',
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=512,
        random_seed=1234,
        tokenizer_max_seq_length=2048,
        **kwargs,
    ):
        if quant_config._requires_modelopt_quantization:
            # modelopt quantization flow
            super().quantize(hf_model_dir,
                             output_dir,
                             dtype=dtype,
                             mapping=mapping,
                             quant_config=quant_config,
                             calib_dataset=calib_dataset,
                             calib_batches=calib_batches,
                             calib_batch_size=calib_batch_size,
                             calib_max_seq_length=calib_max_seq_length,
                             random_seed=random_seed,
                             tokenizer_max_seq_length=tokenizer_max_seq_length)
        elif quant_config._requires_calibration:
            # non-modelopt quantization flow
            from . import convert

            config = QWenConfig.from_hugging_face(hf_model_dir,
                                                  dtype=dtype,
                                                  mapping=mapping,
                                                  quant_config=quant_config,
                                                  **kwargs)
            convert.quantize(hf_model_dir,
                             output_dir,
                             config=config,
                             calib_dataset=calib_dataset)
        else:
            raise ValueError(
                f"The quant_config ({quant_config}) does not require calibration, try {cls.__name__}.from_hugging_face instead."
            )

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)

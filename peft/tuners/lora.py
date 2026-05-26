# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import math
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].
    """
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    adaptive: bool = field(
        default=True,
        metadata={"help": "If True, adaptive k."},
    )
    idx: int = field(
        default=0,
        metadata={"help": "use the idx lora A{idx} and B{idx}"},
    )
    k: int = field(
        default=0,
        metadata={"help": "use top k"},
    )
    universal_idx: Optional[int] = field(
        default=None,
        metadata={"help": "Universal expert index for additive residual routing"},
    )
    shared_lora_a: bool = field(
        default=False,
        metadata={"help": "If True, all lora_A_i across cluster experts share the same weight tensor (HydraFed-LEASE)."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.
    """
    def __init__(self, config: LoraConfig, model: torch.nn.Module):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_8bit or loaded_in_4bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 4-bit/8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "adaptive": self.peft_config.adaptive,
            "idx": self.peft_config.idx,
            "k": self.peft_config.k,
            "universal_idx": self.peft_config.universal_idx,
            "shared_lora_a": self.peft_config.shared_lora_a,
        }

        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                # Handle 8-bit quantization
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update({
                        "has_fp16_weights": target.state.has_fp16_weights,
                        "memory_efficient_backward": target.state.memory_efficient_backward,
                        "threshold": target.state.threshold,
                        "index": target.index,
                    })
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(
                            in_features=target.in_features,
                            out_features=target.out_features,
                            bias=bias,
                            **kwargs
                        )
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear8bitLt(
                            in_features=target.in_features,
                            out_features=target.out_features,
                            bias=bias,
                            **kwargs
                        )

                # Handle 4-bit quantization
                elif loaded_in_4bit and isinstance(target, bnb.nn.Linear4bit):
                    kwargs.update({
                        "compute_dtype": target.compute_dtype,
                        "compress_statistics": target.weight.compress_statistics,
                        "quant_type": target.weight.quant_type,
                    })
                    if self.peft_config.enable_lora is None:
                        new_module = Linear4bit(
                            in_features=target.in_features,
                            out_features=target.out_features,
                            bias=bias,
                            **kwargs
                        )
                    else:
                        kwargs.update({"enable_lora": self.peft_config.enable_lora})
                        new_module = MergedLinear4bit(
                            in_features=target.in_features,
                            out_features=target.out_features,
                            bias=bias,
                            **kwargs
                        )

                # Handle regular Linear layers
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(
                        in_features=target.in_features,
                        out_features=target.out_features,
                        r=self.peft_config.r,
                        lora_alpha=self.peft_config.lora_alpha,
                        lora_nums=self.peft_config.lora_nums,
                        lora_dropout=self.peft_config.lora_dropout,
                        fan_in_fan_out=self.peft_config.fan_in_fan_out,
                        merge_weights=self.peft_config.merge_weights,
                        adaptive=self.peft_config.adaptive,
                        idx=self.peft_config.idx,
                        k=self.peft_config.k,
                        universal_idx=self.peft_config.universal_idx,
                        shared_lora_a=self.peft_config.shared_lora_a,
                    )

                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


def _collapse_route_weight(route_weight: torch.Tensor, lora_num: int, assigned_idx: int) -> torch.Tensor:
    if route_weight.shape[-1] == lora_num:
        return route_weight

    if route_weight.shape[-1] == (2 * lora_num - 1):
        expert_weight = torch.zeros(
            (*route_weight.shape[:-1], lora_num),
            dtype=route_weight.dtype,
            device=route_weight.device,
        )
        expert_weight[..., :lora_num] = route_weight[..., :lora_num]
        expert_weight[..., assigned_idx] += route_weight[..., lora_num:].sum(dim=-1)
        return expert_weight

    raise ValueError(f"Unexpected route dimension {route_weight.shape[-1]} for lora_num={lora_num}")


def _compute_additive_residual_weights(
    route_logits: torch.Tensor,
    lora_num: int,
    assigned_idx: int,
    universal_idx: int,
):
    if universal_idx < 0 or universal_idx >= lora_num:
        raise ValueError(f"universal_idx={universal_idx} is invalid for lora_num={lora_num}")

    route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32)
    expert_weight = _collapse_route_weight(route_weight, lora_num, assigned_idx)

    cluster_mask = torch.ones(lora_num, dtype=torch.bool, device=expert_weight.device)
    cluster_mask[universal_idx] = False
    cluster_weight = expert_weight[..., cluster_mask]
    if cluster_weight.shape[-1] == 0:
        raise ValueError("Additive residual routing requires at least one cluster expert")

    cluster_total = cluster_weight.sum(dim=-1, keepdim=True)
    cluster_weight = cluster_weight / cluster_total.clamp_min(torch.finfo(cluster_weight.dtype).eps)
    cluster_weight = torch.where(cluster_total > 0, cluster_weight, torch.zeros_like(cluster_weight))
    universal_weight = 1.0 - cluster_weight.max(dim=-1, keepdim=True).values

    return cluster_weight, universal_weight


_UEMD_FORWARD_MODES = {
    "full",
    "universal_only",
    "cluster_only",
    "home_cluster_only",
    "non_home_visa",
}


def _apply_uemd_forward_mode(module, cluster_weight: torch.Tensor, universal_weight: torch.Tensor):
    mode = getattr(module, "_uemd_forward_mode", "full")
    if mode == "full":
        return cluster_weight, universal_weight
    if mode == "universal_only":
        return torch.zeros_like(cluster_weight), torch.ones_like(universal_weight)
    if mode == "cluster_only":
        denom = cluster_weight.float().sum(dim=-1, keepdim=True).clamp_min(1e-9)
        cluster_weight = (cluster_weight.float() / denom).to(cluster_weight.dtype)
        return cluster_weight, torch.zeros_like(universal_weight)
    if mode == "home_cluster_only":
        home_local_idx = None
        non_universal = (i for i in range(module.lora_num) if i != module.universal_idx)
        for local_idx, expert_idx in enumerate(non_universal):
            if expert_idx == module.idx:
                home_local_idx = local_idx
                break
        if home_local_idx is None:
            raise ValueError(
                f"home idx {module.idx} not found in non-universal local indices"
            )

        out = torch.zeros_like(cluster_weight)
        out[..., home_local_idx] = 1.0
        return out, torch.zeros_like(universal_weight)
    if mode == "non_home_visa":
        visa_weights = getattr(module, "_visa_cluster_weights", None)
        if visa_weights is None:
            raise RuntimeError(
                "non_home_visa mode requires module._visa_cluster_weights set externally"
            )
        if visa_weights.numel() != cluster_weight.shape[-1]:
            raise ValueError(
                f"visa weight length {visa_weights.numel()} does not match "
                f"cluster weight dimension {cluster_weight.shape[-1]}"
            )

        expanded = visa_weights.to(cluster_weight.device).to(cluster_weight.dtype)
        expanded = expanded.expand(cluster_weight.shape)
        return expanded, torch.zeros_like(universal_weight)
    raise ValueError(f"Unknown UEMD forward mode: {mode!r}")


@contextmanager
def uemd_forward_mode(model: nn.Module, mode: str):
    if mode not in _UEMD_FORWARD_MODES:
        raise ValueError(f"mode must be one of {sorted(_UEMD_FORWARD_MODES)}, got {mode!r}")

    sentinel = object()
    previous_modes = []
    for module in model.modules():
        if getattr(module, "universal_idx", None) is None:
            continue
        previous_modes.append((module, getattr(module, "_uemd_forward_mode", sentinel)))
        module._uemd_forward_mode = mode

    try:
        yield
    finally:
        for module, previous_mode in previous_modes:
            if previous_mode is sentinel:
                if hasattr(module, "_uemd_forward_mode"):
                    delattr(module, "_uemd_forward_mode")
            else:
                module._uemd_forward_mode = previous_mode


class Linear(nn.Linear, LoraLayer):
    """LoRA implemented in a dense layer"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        adaptive: bool = True,
        idx: int = 0,
        k: int = 0,
        universal_idx: Optional[int] = None,
        shared_lora_a: bool = False,
        **kwargs,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_num = lora_nums
        self.adaptive = adaptive
        self.fan_in_fan_out = fan_in_fan_out
        self.idx = idx
        self.k = k
        self.universal_idx = universal_idx
        self.shared_lora_a = shared_lora_a

        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        if r > 0:

            if self.adaptive:
                self.lora_route = nn.Linear(in_features, 2 * self.lora_num - 1, bias=False)
            else:
                self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)

            if self.shared_lora_a:
                # HydraFed-LEASE: all expert paths share the same lora_A weight tensor
                shared_a = nn.Linear(in_features, r, bias=False)
                for i in range(self.lora_num):
                    layer = nn.Linear(in_features, r, bias=False)
                    layer.weight = shared_a.weight  # parameter tying
                    setattr(self, f"lora_A{i}", layer)
                    setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            else:
                for i in range(self.lora_num):
                    setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                    setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        
        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)  
        
        if hasattr(self, 'r') and self.r > 0: 
            for i in range(self.lora_num):
                if hasattr(self, f"lora_A{i}"):
                    nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                if hasattr(self, f"lora_B{i}"):
                    getattr(self, f"lora_B{i}").weight.data.zero_()

            if hasattr(self, "lora_route"):
                nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if hasattr(self, "lora_route"):
            self.lora_route.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").train(mode)
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        if hasattr(self, "lora_route"):
            self.lora_route.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_A{i}").eval()
            getattr(self, f"lora_B{i}").eval()

    def forward(self, x: torch.Tensor):
        weight = self.weight

        if self.disable_adapters:
            result = F.linear(x, transpose(weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(weight, self.fan_in_fan_out), bias=self.bias)

            if self.r > 0:
                route_logits = self.lora_route(x)

                dropped_x = self.lora_dropout(x)

                if self.universal_idx is not None:
                    cluster_weight, universal_weight = _compute_additive_residual_weights(
                        route_logits,
                        self.lora_num,
                        self.idx,
                        self.universal_idx,
                    )
                    cluster_weight = cluster_weight.to(result.dtype)
                    universal_weight = universal_weight.to(result.dtype)
                    cluster_weight, universal_weight = _apply_uemd_forward_mode(
                        self,
                        cluster_weight,
                        universal_weight,
                    )

                    for local_idx, expert_idx in enumerate(i for i in range(self.lora_num) if i != self.universal_idx):
                        lora_A = getattr(self, f"lora_A{expert_idx}")
                        lora_B = getattr(self, f"lora_B{expert_idx}")
                        expert_output = lora_B(lora_A(dropped_x))
                        result = result + torch.unsqueeze(cluster_weight[..., local_idx], -1) * expert_output * self.scaling

                    universal_A = getattr(self, f"lora_A{self.universal_idx}")
                    universal_B = getattr(self, f"lora_B{self.universal_idx}")
                    universal_output = universal_B(universal_A(dropped_x))
                    result = result + universal_weight * universal_output * self.scaling
                    blcls = torch.zeros(1, dtype=result.dtype, device=result.device)[0]
                    return result

                if self.adaptive:
                    route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32)
                    route_weight = route_weight.to(result.dtype)

                    top_weights, top_indices = torch.topk(route_weight, self.lora_num, dim=-1)

                    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

                    batch_size = x.shape[0]
                    seq_len = x.shape[1] if x.dim() > 2 else 1

                    module_mapping = torch.arange(route_weight.shape[-1])
                    module_mapping[self.lora_num:] = self.idx 

                    for idx_k in range(self.lora_num):
                        route_dim_indices = top_indices[:, :, idx_k]

                        expert_indices = module_mapping.to(route_dim_indices.device)[route_dim_indices]

                        expert_weights = top_weights[:, :, idx_k]
                        expert_weights = torch.unsqueeze(expert_weights, -1)

                        for i in range(self.lora_num):
                            mask = (expert_indices == i)
                            if not mask.any():
                                continue

                            lora_A = getattr(self, f"lora_A{i}")
                            lora_B = getattr(self, f"lora_B{i}")

                            lora_A_output = lora_A(dropped_x)
                            expert_output = lora_B(lora_A_output)

                            masked_weights = torch.zeros_like(expert_weights)
                            masked_weights[mask] = expert_weights[mask]

                            result = result + masked_weights * expert_output * self.scaling

                else:
                    route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32)

                    k = min(self.k, self.lora_num) if self.k > 0 else self.lora_num

                    if k < self.lora_num:
                        top_k_weights, top_k_indices = torch.topk(route_weight, k, dim=-1)

                        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

                        top_k_weights = top_k_weights.to(result.dtype)

                        for idx_k in range(k):
                            expert_indices = top_k_indices[:, :, idx_k]

                            expert_weights = top_k_weights[:, :, idx_k]
                            expert_weights = torch.unsqueeze(expert_weights, -1)

                            for i in range(self.lora_num):
                                mask = (expert_indices == i)
                                if not mask.any():
                                    continue

                                lora_A = getattr(self, f"lora_A{i}")
                                lora_B = getattr(self, f"lora_B{i}")

                                lora_A_output = lora_A(dropped_x)
                                expert_output = lora_B(lora_A_output)

                                masked_weights = torch.zeros_like(expert_weights)
                                masked_weights[mask] = expert_weights[mask]

                                result = result + masked_weights * expert_output * self.scaling
                    else:
                        route_weight = route_weight.to(result.dtype)

                        for i in range(self.lora_num):
                            lora_A = getattr(self, f"lora_A{i}")
                            lora_B = getattr(self, f"lora_B{i}")
                            lora_A_output = lora_A(dropped_x)
                            scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                            lora_output = lora_B(lora_A_output)
                            result = result + scaled_route * lora_output * self.scaling

            blcls = torch.zeros(1, dtype=result.dtype, device=result.device)[0]
            return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
        

class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
    def __init__(
        self,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        adaptive: bool = False,
        idx: int = 0,
        k: int = 0,
        universal_idx: Optional[int] = None,
        shared_lora_a: bool = False,
        **kwargs,
    ):
        bnb.nn.Linear8bitLt.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            has_fp16_weights=kwargs.get("has_fp16_weights", True),
            memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
            threshold=kwargs.get("threshold", 0.0),
            index=kwargs.get("index", None),
        )
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)

        self.lora_num = lora_nums
        self.adaptive = adaptive
        self.idx = idx
        self.k = k
        self.universal_idx = universal_idx
        self.shared_lora_a = shared_lora_a

        if r > 0:
            if self.adaptive:
                self.lora_route = nn.Linear(in_features, 2 * self.lora_num - 1, bias=False)
            else:
                self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)

            if self.shared_lora_a:
                shared_a = nn.Linear(in_features, r, bias=False)
                for i in range(self.lora_num):
                    layer = nn.Linear(in_features, r, bias=False)
                    layer.weight = shared_a.weight
                    setattr(self, f"lora_A{i}", layer)
                    setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
            else:
                for i in range(self.lora_num):
                    setattr(self, f"lora_A{i}", nn.Linear(in_features, r, bias=False))
                    setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))
                
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'r') and self.r > 0:
            for i in range(self.lora_num):
                if hasattr(self, f"lora_A{i}"):
                    nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                if hasattr(self, f"lora_B{i}"):
                    getattr(self, f"lora_B{i}").weight.data.zero_()
            if hasattr(self, "lora_route"):
                nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters:
            return result
        elif self.r > 0:
            def process_with_routing(x_input, target_dtype=None):
                route_logits = self.lora_route(x_input)
                
                dropped_x = self.lora_dropout(x_input)

                if self.universal_idx is not None:
                    cluster_weight, universal_weight = _compute_additive_residual_weights(
                        route_logits,
                        self.lora_num,
                        self.idx,
                        self.universal_idx,
                    )
                    if target_dtype is not None:
                        cluster_weight = cluster_weight.to(target_dtype)
                        universal_weight = universal_weight.to(target_dtype)
                    cluster_weight, universal_weight = _apply_uemd_forward_mode(
                        self,
                        cluster_weight,
                        universal_weight,
                    )

                    current_result = result
                    for local_idx, expert_idx in enumerate(i for i in range(self.lora_num) if i != self.universal_idx):
                        lora_A = getattr(self, f"lora_A{expert_idx}")
                        lora_B = getattr(self, f"lora_B{expert_idx}")
                        expert_output = lora_B(lora_A(dropped_x))
                        current_result = (
                            current_result
                            + torch.unsqueeze(cluster_weight[..., local_idx], -1) * expert_output * self.scaling
                        )

                    universal_A = getattr(self, f"lora_A{self.universal_idx}")
                    universal_B = getattr(self, f"lora_B{self.universal_idx}")
                    universal_output = universal_B(universal_A(dropped_x))
                    current_result = current_result + universal_weight * universal_output * self.scaling
                    return current_result
                
                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32)
                if target_dtype is not None:
                    route_weight = route_weight.to(target_dtype)
                
                current_result = result
                
                if self.adaptive:                    
                    top_weights, top_indices = torch.topk(route_weight, self.lora_num, dim=-1)
                    
                    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
                    if target_dtype is not None:
                        top_weights = top_weights.to(target_dtype)
                    
                    module_mapping = torch.arange(route_weight.shape[-1])
                    module_mapping[self.lora_num:] = self.idx  
                    
                    for idx_k in range(self.lora_num):
                        route_dim_indices = top_indices[:, :, idx_k]
                        
                        expert_indices = module_mapping.to(route_dim_indices.device)[route_dim_indices]
                        
                        expert_weights = top_weights[:, :, idx_k]
                        expert_weights = torch.unsqueeze(expert_weights, -1)
                        
                        for i in range(self.lora_num):
                            mask = (expert_indices == i)
                            if not mask.any():
                                continue
                            
                            lora_A = getattr(self, f"lora_A{i}")
                            lora_B = getattr(self, f"lora_B{i}")
                            
                            lora_A_output = lora_A(dropped_x)
                            expert_output = lora_B(lora_A_output)
                            
                            masked_weights = torch.zeros_like(expert_weights)
                            masked_weights[mask] = expert_weights[mask]
                            
                            current_result = current_result + masked_weights * expert_output * self.scaling
                
                else:
                    k = min(self.k, self.lora_num) if self.k > 0 else self.lora_num
                    
                    if k < self.lora_num:
                        top_k_weights, top_k_indices = torch.topk(route_weight, k, dim=-1)
                        
                        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
                        if target_dtype is not None:
                            top_k_weights = top_k_weights.to(target_dtype)
                        
                        for idx_k in range(k):
                            expert_indices = top_k_indices[:, :, idx_k]
                            
                            expert_weights = top_k_weights[:, :, idx_k]
                            expert_weights = torch.unsqueeze(expert_weights, -1)

                            for i in range(self.lora_num):
                                mask = (expert_indices == i)
                                if not mask.any():
                                    continue
                                
                                lora_A = getattr(self, f"lora_A{i}")
                                lora_B = getattr(self, f"lora_B{i}")
                                
                                lora_A_output = lora_A(dropped_x)
                                expert_output = lora_B(lora_A_output)
                                
                                masked_weights = torch.zeros_like(expert_weights)
                                masked_weights[mask] = expert_weights[mask]
                                
                                current_result = current_result + masked_weights * expert_output * self.scaling
                    else:
                        for i in range(self.lora_num):
                            lora_A = getattr(self, f"lora_A{i}")
                            lora_B = getattr(self, f"lora_B{i}")
                            lora_A_output = lora_A(dropped_x)
                            scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                            lora_output = lora_B(lora_A_output)
                            current_result = current_result + scaled_route * lora_output * self.scaling
                
                return current_result
            
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()
                
                result = process_with_routing(x, expected_dtype)
                result = result.to(expected_dtype)
            else:
                result = process_with_routing(x)
        
        blcls = torch.zeros(1, dtype=result.dtype, device=result.device)[0]
        return result


class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        adaptive: bool = False,
        idx: int = 0,
        k: int = 0,
        universal_idx: Optional[int] = None,
        **kwargs,
    ):
        bnb.nn.Linear8bitLt.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            has_fp16_weights=kwargs.get("has_fp16_weights", True),
            memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
            threshold=kwargs.get("threshold", 0.0),
            index=kwargs.get("index", None),
        )
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
        
        if out_features % len(enable_lora) != 0:
            raise ValueError("The length of enable_lora must divide out_features")
            
        self.enable_lora = enable_lora
        self.lora_num = lora_nums
        self.adaptive = adaptive
        self.idx = idx
        self.k = k
        self.universal_idx = universal_idx
        
        if r > 0 and any(enable_lora):
            if self.adaptive:
                self.lora_route = nn.Linear(in_features, 2 * self.lora_num - 1, bias=False)
            else:
                self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
            
            for i in range(self.lora_num):
                setattr(self, f"lora_A{i}", nn.Linear(in_features, r * sum(enable_lora), bias=False))
                setattr(self, f"lora_B{i}", nn.Conv1d(
                    r * sum(enable_lora),
                    out_features // len(enable_lora) * sum(enable_lora),
                    kernel_size=1,
                    groups=sum(enable_lora),
                    bias=False
                ))
            
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'r') and self.r > 0:
            for i in range(self.lora_num):
                if hasattr(self, f"lora_A{i}"):
                    nn.init.kaiming_uniform_(getattr(self, f"lora_A{i}").weight, a=math.sqrt(5))
                if hasattr(self, f"lora_B{i}"):
                    getattr(self, f"lora_B{i}").weight.data.zero_()
            if hasattr(self, "lora_route"):
                nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def forward(self, x: torch.Tensor, task_types=None):
        result = super().forward(x)
        
        if self.disable_adapters:
            return result
        elif self.r > 0:
            def process_with_routing(x_input, target_dtype=None):
                
                route_logits = self.lora_route(x_input)
                
                dropped_x = self.lora_dropout(x_input)

                if self.universal_idx is not None:
                    cluster_weight, universal_weight = _compute_additive_residual_weights(
                        route_logits,
                        self.lora_num,
                        self.idx,
                        self.universal_idx,
                    )
                    if target_dtype is not None:
                        cluster_weight = cluster_weight.to(target_dtype)
                        universal_weight = universal_weight.to(target_dtype)
                    cluster_weight, universal_weight = _apply_uemd_forward_mode(
                        self,
                        cluster_weight,
                        universal_weight,
                    )

                    current_result = result
                    for local_idx, expert_idx in enumerate(i for i in range(self.lora_num) if i != self.universal_idx):
                        lora_A = getattr(self, f"lora_A{expert_idx}")
                        lora_B = getattr(self, f"lora_B{expert_idx}")
                        after_A = lora_A(dropped_x).transpose(-2, -1)
                        after_B = lora_B(after_A).transpose(-2, -1)

                        if target_dtype is not None:
                            output = self.zero_pad(after_B).to(target_dtype) * self.scaling
                        else:
                            output = self.zero_pad(after_B) * self.scaling

                        current_result = current_result + torch.unsqueeze(cluster_weight[..., local_idx], -1) * output

                    universal_A = getattr(self, f"lora_A{self.universal_idx}")
                    universal_B = getattr(self, f"lora_B{self.universal_idx}")
                    universal_after_A = universal_A(dropped_x).transpose(-2, -1)
                    universal_after_B = universal_B(universal_after_A).transpose(-2, -1)

                    if target_dtype is not None:
                        universal_output = self.zero_pad(universal_after_B).to(target_dtype) * self.scaling
                    else:
                        universal_output = self.zero_pad(universal_after_B) * self.scaling

                    current_result = current_result + universal_weight * universal_output
                    return current_result
                
                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32)
                if target_dtype is not None:
                    route_weight = route_weight.to(target_dtype)
                
                current_result = result
                
                if self.adaptive:
                    top_weights, top_indices = torch.topk(route_weight, self.lora_num, dim=-1)
                    
                    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
                    if target_dtype is not None:
                        top_weights = top_weights.to(target_dtype)
                    
                    module_mapping = torch.arange(route_weight.shape[-1])
                    module_mapping[self.lora_num:] = self.idx  
                    
                    for idx_k in range(self.lora_num):
                        route_dim_indices = top_indices[:, :, idx_k]
                        
                        expert_indices = module_mapping.to(route_dim_indices.device)[route_dim_indices]
                        
                        expert_weights = top_weights[:, :, idx_k]
                        expert_weights = torch.unsqueeze(expert_weights, -1)
                        
                        for i in range(self.lora_num):
                            mask = (expert_indices == i)
                            if not mask.any():
                                continue
                            
                            lora_A = getattr(self, f"lora_A{i}")
                            lora_B = getattr(self, f"lora_B{i}")
                            
                            after_A = lora_A(dropped_x).transpose(-2, -1)
                            after_B = lora_B(after_A).transpose(-2, -1)
                            
                            if target_dtype is not None:
                                output = self.zero_pad(after_B).to(target_dtype) * self.scaling
                            else:
                                output = self.zero_pad(after_B) * self.scaling
                            
                            masked_weights = torch.zeros_like(expert_weights)
                            masked_weights[mask] = expert_weights[mask]
                            
                            current_result = current_result + masked_weights * output
                
                else:
                    k = min(self.k, self.lora_num) if self.k > 0 else self.lora_num
                    
                    if k < self.lora_num:
                        top_k_weights, top_k_indices = torch.topk(route_weight, k, dim=-1)
                        
                        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
                        if target_dtype is not None:
                            top_k_weights = top_k_weights.to(target_dtype)
                        
                        for idx_k in range(k):
                            expert_indices = top_k_indices[:, :, idx_k]
                            
                            expert_weights = top_k_weights[:, :, idx_k]
                            expert_weights = torch.unsqueeze(expert_weights, -1)
                            
                            for i in range(self.lora_num):
                                mask = (expert_indices == i)
                                if not mask.any():
                                    continue
                                
                                lora_A = getattr(self, f"lora_A{i}")
                                lora_B = getattr(self, f"lora_B{i}")
                                
                                after_A = lora_A(dropped_x).transpose(-2, -1)
                                after_B = lora_B(after_A).transpose(-2, -1)
                                
                                if target_dtype is not None:
                                    output = self.zero_pad(after_B).to(target_dtype) * self.scaling
                                else:
                                    output = self.zero_pad(after_B) * self.scaling
                                
                                masked_weights = torch.zeros_like(expert_weights)
                                masked_weights[mask] = expert_weights[mask]
                                
                                current_result = current_result + masked_weights * output
                    else:
                        for i in range(self.lora_num):
                            lora_A = getattr(self, f"lora_A{i}")
                            lora_B = getattr(self, f"lora_B{i}")
                            after_A = lora_A(dropped_x).transpose(-2, -1)
                            scaled_route = torch.unsqueeze(route_weight[:, :, i], -1)
                            after_B = lora_B(after_A).transpose(-2, -1)
                            
                            if target_dtype is not None:
                                output = self.zero_pad(after_B).to(target_dtype) * self.scaling
                            else:
                                output = self.zero_pad(after_B) * self.scaling
                                
                            current_result = current_result + scaled_route * output
                
                return current_result
            
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()
                
                result = process_with_routing(x, expected_dtype)
                result = result.to(expected_dtype)
            else:
                result = process_with_routing(x)
        
        blcls = torch.zeros(1, dtype=result.dtype, device=result.device)[0]
        return result
    
    

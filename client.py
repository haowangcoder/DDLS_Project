import os
import torch
import json
import numpy as np
import gc
import re
import copy
from contextlib import nullcontext
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import uemd_forward_mode


def _collapse_expert_probs(route_weight: torch.Tensor, lora_n: int, assigned_idx: int | None) -> torch.Tensor:
    if route_weight.shape[-1] == lora_n:
        return route_weight

    if route_weight.shape[-1] == (2 * lora_n - 1):
        if assigned_idx is None:
            raise ValueError("assigned_idx is required to collapse adaptive routing probabilities")

        expert_weight = torch.zeros(
            (*route_weight.shape[:-1], lora_n),
            dtype=route_weight.dtype,
            device=route_weight.device,
        )
        expert_weight[..., :lora_n] = route_weight[..., :lora_n]
        expert_weight[..., assigned_idx] += route_weight[..., lora_n:].sum(dim=-1)
        return expert_weight

    raise ValueError(f"Unexpected route dimension {route_weight.shape[-1]} for lora_n={lora_n}")


def _collapse_additive_residual_probs(
    route_weight: torch.Tensor,
    lora_n: int,
    assigned_idx: int | None,
    universal_idx: int,
) -> torch.Tensor:
    expert_weight = _collapse_expert_probs(route_weight, lora_n, assigned_idx)

    if universal_idx < 0 or universal_idx >= lora_n:
        raise ValueError(f"universal_idx={universal_idx} is invalid for lora_n={lora_n}")

    cluster_mask = torch.ones(lora_n, dtype=torch.bool, device=expert_weight.device)
    cluster_mask[universal_idx] = False
    cluster_weight = expert_weight[..., cluster_mask]
    cluster_total = cluster_weight.sum(dim=-1, keepdim=True)
    cluster_weight = cluster_weight / cluster_total.clamp_min(torch.finfo(cluster_weight.dtype).eps)
    cluster_weight = torch.where(cluster_total > 0, cluster_weight, torch.zeros_like(cluster_weight))
    universal_weight = 1.0 - cluster_weight.max(dim=-1, keepdim=True).values

    effective_weight = torch.zeros_like(expert_weight)
    effective_weight[..., cluster_mask] = cluster_weight
    effective_weight[..., universal_idx] = universal_weight.squeeze(-1)
    return effective_weight


def _topk_expert_mask(route_weight: torch.Tensor, lora_n: int, assigned_idx: int | None) -> torch.Tensor:
    top_indices = torch.topk(route_weight, min(lora_n, route_weight.shape[-1]), dim=-1).indices

    if route_weight.shape[-1] == lora_n:
        expert_indices = top_indices
    elif route_weight.shape[-1] == (2 * lora_n - 1):
        if assigned_idx is None:
            raise ValueError("assigned_idx is required to collapse adaptive routing assignments")

        module_mapping = torch.arange(route_weight.shape[-1], device=route_weight.device)
        module_mapping[lora_n:] = assigned_idx
        expert_indices = module_mapping[top_indices]
    else:
        raise ValueError(f"Unexpected route dimension {route_weight.shape[-1]} for lora_n={lora_n}")

    return F.one_hot(expert_indices, num_classes=lora_n).sum(dim=-2).clamp(max=1).to(route_weight.dtype)


def _build_trainable_experts(
    client_lora_group: int | None,
    universal_idx: int | None,
    soft_membership_for_client: dict[int, float] | None = None,
    threshold: float = 0.0,
) -> set[int]:
    if soft_membership_for_client:
        trainable_experts = {
            int(expert_idx)
            for expert_idx, weight in soft_membership_for_client.items()
            if weight > threshold
        }
    else:
        trainable_experts = {client_lora_group} if client_lora_group is not None else set()
    if universal_idx is not None:
        trainable_experts.add(universal_idx)
    return trainable_experts


class LoadBalancedTrainer(Trainer):
    def __init__(self, *args, load_balance_coeff=0.01, lora_n=4, assigned_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_balance_coeff = load_balance_coeff
        self.lora_n = lora_n
        self.assigned_idx = assigned_idx
        self._route_snapshots = []
        self._hooks = []
        self._install_hooks()

    def _install_hooks(self):
        self.close()

        for name, module in self.model.named_modules():
            if name.endswith("lora_route"):
                self._hooks.append(module.register_forward_hook(self._capture_route))

    def _capture_route(self, _module, _inputs, output):
        self._route_snapshots.append((self.assigned_idx, output))

    def _compute_load_balance_loss(self):
        if not self._route_snapshots or self.load_balance_coeff <= 0:
            return None

        all_probs = []
        all_assignments = []

        for assigned_idx, logits in self._route_snapshots:
            probs = torch.softmax(logits.float(), dim=-1)
            expert_probs = _collapse_expert_probs(probs, self.lora_n, assigned_idx)
            expert_assignments = _topk_expert_mask(probs, self.lora_n, assigned_idx)

            all_probs.append(expert_probs.reshape(-1, expert_probs.shape[-1]))
            all_assignments.append(expert_assignments.reshape(-1, expert_assignments.shape[-1]))

        if not all_probs:
            return None

        all_probs = torch.cat(all_probs, dim=0)
        all_assignments = torch.cat(all_assignments, dim=0)

        if all_probs.numel() == 0 or all_assignments.numel() == 0:
            return None

        f = all_assignments.mean(dim=0)
        p = all_probs.mean(dim=0)
        return self.load_balance_coeff * self.lora_n * torch.sum(f * p)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._route_snapshots = []
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        load_balance_loss = self._compute_load_balance_loss()
        if load_balance_loss is not None:
            loss = loss + load_balance_loss

        return (loss, outputs) if return_outputs else loss

    def close(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


class UEMDTrainer(Trainer):
    def __init__(self, *args, uemd_coeff=0.0, lora_n=4, universal_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.uemd_coeff = float(uemd_coeff)
        self.lora_n = int(lora_n)
        self.universal_idx = universal_idx
        self._hooks = []
        self._module_to_meta = {}
        self._partial_outputs = {}
        self._uemd_partial_losses = []

        if self.uemd_coeff > 0:
            if self.universal_idx is None:
                raise ValueError("UEMD requires universal_idx to identify the universal expert")
            self.universal_idx = int(self.universal_idx)
            if self.lora_n < 2:
                raise ValueError("UEMD requires at least one cluster expert and one universal expert")
            if self.universal_idx < 0 or self.universal_idx >= self.lora_n:
                raise ValueError(f"universal_idx={self.universal_idx} is invalid for lora_n={self.lora_n}")
            self._install_hooks()

    def _install_hooks(self):
        self.close()
        self._module_to_meta = {}

        modules = dict(self.model.named_modules())
        for name, module in modules.items():
            match = re.search(r"(?:^|\.)lora_B(\d+)$", name)
            if match is None:
                continue

            expert_idx = int(match.group(1))
            if expert_idx >= self.lora_n:
                continue

            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = modules.get(parent_name)
            if parent is None:
                continue

            self._module_to_meta[id(module)] = (id(parent), expert_idx)
            self._hooks.append(module.register_forward_hook(self._capture_lora_b_output))

    def _capture_lora_b_output(self, module, _inputs, output):
        if self.uemd_coeff <= 0 or not torch.is_tensor(output):
            return

        meta = self._module_to_meta.get(id(module))
        if meta is None:
            return

        parent_id, expert_idx = meta
        parent_outputs = self._partial_outputs.setdefault(parent_id, {})
        parent_outputs[expert_idx] = output if expert_idx == self.universal_idx else output.detach()

        if len(parent_outputs) != self.lora_n:
            return

        outputs = self._partial_outputs.pop(parent_id)
        universal_output = outputs.get(self.universal_idx)
        if universal_output is None:
            return

        cluster_outputs = [
            outputs[idx]
            for idx in range(self.lora_n)
            if idx != self.universal_idx and idx in outputs
        ]
        if len(cluster_outputs) != self.lora_n - 1:
            return

        cluster_mean = torch.stack([out.detach() for out in cluster_outputs], dim=0).mean(dim=0)
        cluster_mean = cluster_mean.to(device=universal_output.device, dtype=universal_output.dtype)
        self._uemd_partial_losses.append(((universal_output - cluster_mean) ** 2).mean())

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._partial_outputs = {}
        self._uemd_partial_losses = []
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        if self.uemd_coeff > 0 and self._uemd_partial_losses:
            uemd_loss = torch.stack(self._uemd_partial_losses).mean()
            loss = loss + self.uemd_coeff * uemd_loss

        return (loss, outputs) if return_outputs else loss

    def close(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._module_to_meta = {}


class LogitUEMDTrainer(Trainer):
    def __init__(self, *args, uemd_logit_coeff=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.uemd_logit_coeff = float(uemd_logit_coeff)
        self.uemd_model = self.model

    @staticmethod
    def _extract_loss(outputs):
        if isinstance(outputs, dict):
            return outputs["loss"]
        return outputs.loss if hasattr(outputs, "loss") else outputs[0]

    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, dict):
            return outputs["logits"]
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[1] if len(outputs) > 1 else outputs[0]

    @staticmethod
    def _without_labels(inputs):
        label_keys = {"labels", "label", "label_ids"}
        return {key: value for key, value in inputs.items() if key not in label_keys}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.uemd_logit_coeff <= 0:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        outputs = model(**inputs)
        loss = self._extract_loss(outputs)
        aux_inputs = self._without_labels(inputs)

        with uemd_forward_mode(model, "universal_only"):
            univ_outputs = model(**aux_inputs)
        univ_logits = self._extract_logits(univ_outputs)

        with torch.no_grad():
            with uemd_forward_mode(model, "cluster_only"):
                clust_outputs = model(**aux_inputs)
        clust_logits = self._extract_logits(clust_outputs)

        uemd_logit_loss = F.kl_div(
            F.log_softmax(univ_logits, dim=-1),
            F.softmax(clust_logits.detach(), dim=-1),
            reduction="batchmean",
        )
        loss = loss + self.uemd_logit_coeff * uemd_logit_loss

        return (loss, outputs) if return_outputs else loss


class WriteOnlySoftTrainer(Trainer):
    _RDROP_DIRECTIONS = ("symmetric", "home_to_visa", "visa_to_home")
    _RDROP_STOPGRAD_MODES = ("target", "none", "both")

    def __init__(
        self,
        *args,
        visa_coeff=0.0,
        soft_membership_for_client=None,
        lora_n=4,
        universal_idx=None,
        home_idx=None,
        affinity_mode="off",
        rdrop_kl_coeff=0.0,
        rdrop_direction="symmetric",
        rdrop_stopgrad="target",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.visa_coeff = float(visa_coeff)
        self.soft_membership_for_client = {
            int(expert_idx): float(weight)
            for expert_idx, weight in (soft_membership_for_client or {}).items()
        }
        self.lora_n = int(lora_n)
        self.universal_idx = universal_idx
        self.home_idx = home_idx
        self.affinity_mode = affinity_mode
        self._visa_weights_set = False

        self.rdrop_kl_coeff = float(rdrop_kl_coeff)
        if rdrop_direction not in self._RDROP_DIRECTIONS:
            raise ValueError(
                f"rdrop_direction must be one of {self._RDROP_DIRECTIONS}, got {rdrop_direction!r}"
            )
        if rdrop_stopgrad not in self._RDROP_STOPGRAD_MODES:
            raise ValueError(
                f"rdrop_stopgrad must be one of {self._RDROP_STOPGRAD_MODES}, got {rdrop_stopgrad!r}"
            )
        self.rdrop_direction = rdrop_direction
        self.rdrop_stopgrad = rdrop_stopgrad

    @staticmethod
    def _extract_loss(outputs):
        if isinstance(outputs, dict):
            return outputs["loss"]
        return outputs.loss if hasattr(outputs, "loss") else outputs[0]

    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, dict):
            return outputs["logits"]
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[1] if len(outputs) > 1 else outputs[0]

    @staticmethod
    def _without_labels(inputs):
        label_keys = {"labels", "label", "label_ids"}
        return {key: value for key, value in inputs.items() if key not in label_keys}

    def _compute_rdrop_kl(self, logits_home, logits_visa):
        """Symmetric / one-sided KL between home and visa output distributions.

        stopgrad policy:
          target — detach the right-hand side of each KL (recommended; keeps the
                   visa branch from being pulled toward home and killing exposure)
          none   — no stop-grad (matches the original R-Drop formulation)
          both   — detach both sides (regularizer treated as fully external)
        """
        if self.rdrop_stopgrad == "both":
            logits_home_lhs = logits_home.detach()
            logits_visa_lhs = logits_visa.detach()
        else:
            logits_home_lhs = logits_home
            logits_visa_lhs = logits_visa

        if self.rdrop_stopgrad in ("target", "both"):
            target_home = logits_home.detach()
            target_visa = logits_visa.detach()
        else:
            target_home = logits_home
            target_visa = logits_visa

        kl_terms = []
        if self.rdrop_direction in ("symmetric", "home_to_visa"):
            kl_terms.append(
                F.kl_div(
                    F.log_softmax(logits_home_lhs, dim=-1),
                    F.softmax(target_visa, dim=-1),
                    reduction="batchmean",
                )
            )
        if self.rdrop_direction in ("symmetric", "visa_to_home"):
            kl_terms.append(
                F.kl_div(
                    F.log_softmax(logits_visa_lhs, dim=-1),
                    F.softmax(target_home, dim=-1),
                    reduction="batchmean",
                )
            )

        if len(kl_terms) == 1:
            return kl_terms[0]
        return 0.5 * (kl_terms[0] + kl_terms[1])

    def collect_home_signature(self, model, num_batches=8):
        if self.affinity_mode == "off":
            return None

        max_batches = int(num_batches)
        if max_batches <= 0:
            return None

        sequence_model = _get_sequence_classifier_model(model)
        backbone = _get_fedrod_backbone(sequence_model)
        dataloader = self.get_train_dataloader()

        was_training = model.training
        model.eval()
        signature_sum = None
        example_count = 0

        try:
            with torch.no_grad():
                for batch_idx, inputs in enumerate(dataloader):
                    if batch_idx >= max_batches:
                        break

                    inputs = self._prepare_inputs(inputs)
                    signature_inputs = self._without_labels(inputs)
                    with uemd_forward_mode(model, "home_cluster_only"):
                        outputs = backbone(**signature_inputs, return_dict=True)

                    cls_hidden = outputs.last_hidden_state[:, 0, :].detach().float().cpu()
                    batch_size = cls_hidden.shape[0]
                    if signature_sum is None:
                        signature_sum = cls_hidden.sum(dim=0)
                    else:
                        signature_sum += cls_hidden.sum(dim=0)
                    example_count += batch_size
        finally:
            if was_training:
                model.train()

        if signature_sum is None or example_count == 0:
            return None
        return signature_sum / example_count

    def refresh_visa_weights(self, new_soft_membership: dict[int, float]):
        if self.affinity_mode == "off" or new_soft_membership is None:
            return

        self.soft_membership_for_client = {
            int(expert_idx): float(weight)
            for expert_idx, weight in new_soft_membership.items()
        }
        # The next forward call will lazily rebuild module._visa_cluster_weights.
        self._visa_weights_set = False

    def _ensure_visa_weights(self, model):
        if self._visa_weights_set or self.visa_coeff <= 0:
            return
        if self.universal_idx is None:
            raise ValueError("write-only soft training requires universal_idx")
        if self.home_idx is None:
            raise ValueError("write-only soft training requires home_idx")

        universal_idx = int(self.universal_idx)
        home_idx = int(self.home_idx)
        non_universal = [i for i in range(self.lora_n) if i != universal_idx]
        if home_idx not in non_universal:
            raise ValueError(
                f"home_idx={home_idx} is not a non-universal expert for lora_n={self.lora_n}"
            )

        weights = []
        for expert_idx in non_universal:
            if expert_idx == home_idx:
                weights.append(0.0)
            else:
                weights.append(float(self.soft_membership_for_client.get(expert_idx, 0.0)))

        total = sum(weights)
        if total <= 1e-12:
            n_non_home = max(1, len(non_universal) - 1)
            weights = [
                0.0 if expert_idx == home_idx else 1.0 / n_non_home
                for expert_idx in non_universal
            ]
        else:
            weights = [weight / total for weight in weights]

        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        attached = 0
        for module in model.modules():
            if getattr(module, "universal_idx", None) is None:
                continue
            module_device = next(module.parameters()).device
            module._visa_cluster_weights = weight_tensor.clone().to(module_device)
            attached += 1
        if attached == 0:
            raise RuntimeError("write-only soft training found no universal LoRA modules")
        self._visa_weights_set = True

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._ensure_visa_weights(model)

        with uemd_forward_mode(model, "home_cluster_only"):
            outputs_home = model(**inputs)
        home_loss = self._extract_loss(outputs_home)

        if self.visa_coeff <= 0:
            return (home_loss, outputs_home) if return_outputs else home_loss

        with uemd_forward_mode(model, "non_home_visa"):
            outputs_visa = model(**inputs)
        visa_loss = self._extract_loss(outputs_visa)

        total = home_loss + self.visa_coeff * visa_loss

        if self.rdrop_kl_coeff > 0:
            logits_home = self._extract_logits(outputs_home)
            logits_visa = self._extract_logits(outputs_visa)
            kl = self._compute_rdrop_kl(logits_home, logits_visa)
            total = total + self.rdrop_kl_coeff * kl

        return (total, outputs_home) if return_outputs else total


def _get_sequence_classifier_model(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def _get_fedrod_backbone(sequence_model):
    backbone_name = getattr(sequence_model, "base_model_prefix", None)
    if backbone_name and hasattr(sequence_model, backbone_name):
        return getattr(sequence_model, backbone_name)
    if hasattr(sequence_model, "roberta"):
        return sequence_model.roberta
    raise NotImplementedError(
        "FedRoD dual-head forward currently expects a Hugging Face sequence "
        "classifier with a named transformer backbone, e.g. RobertaForSequenceClassification"
    )


def ensure_fedrod_dual_head(model, lora_n, universal_idx, alpha_init=2.0):
    if universal_idx is None:
        raise ValueError("--fedrod_dual_head requires a universal expert index")
    if lora_n < 2:
        raise ValueError("--fedrod_dual_head requires at least one cluster expert and one universal expert")

    sequence_model = _get_sequence_classifier_model(model)
    if not hasattr(sequence_model, "classifier"):
        raise NotImplementedError("FedRoD dual-head currently expects a model.classifier head")

    if not hasattr(model, "cls_universal"):
        model.cls_universal = copy.deepcopy(sequence_model.classifier)

    expected_alpha_shape = (int(lora_n) - 1,)
    if not hasattr(model, "alpha_logit") or tuple(model.alpha_logit.shape) != expected_alpha_shape:
        reference_param = next(model.cls_universal.parameters())
        alpha = torch.full(
            expected_alpha_shape,
            float(alpha_init),
            device=reference_param.device,
            dtype=torch.float32,
        )
        model.alpha_logit = nn.Parameter(alpha)

    model.fedrod_lora_n = int(lora_n)
    model.fedrod_universal_idx = int(universal_idx)
    return model


def _fedrod_alpha(model, home_cluster_idx):
    lora_n = int(getattr(model, "fedrod_lora_n", 0))
    universal_idx = int(getattr(model, "fedrod_universal_idx", -1))
    if lora_n <= 1 or universal_idx < 0:
        raise ValueError("FedRoD dual-head model is missing lora_n/universal_idx metadata")
    if home_cluster_idx is None:
        raise ValueError("FedRoD dual-head forward requires home_cluster_idx")

    non_universal = [idx for idx in range(lora_n) if idx != universal_idx]
    try:
        alpha_idx = non_universal.index(int(home_cluster_idx))
    except ValueError as exc:
        raise ValueError(
            f"home_cluster_idx={home_cluster_idx} is not a non-universal expert "
            f"for lora_n={lora_n}, universal_idx={universal_idx}"
        ) from exc
    return torch.sigmoid(model.alpha_logit[alpha_idx])


def fedrod_mix_logits(model, cluster_logits, universal_logits, home_cluster_idx):
    alpha = _fedrod_alpha(model, home_cluster_idx).to(
        device=cluster_logits.device,
        dtype=cluster_logits.dtype,
    )
    return alpha * cluster_logits + (1.0 - alpha) * universal_logits


def _sequence_classification_loss(config, logits, labels):
    if labels is None:
        return None

    num_labels = int(getattr(config, "num_labels", logits.shape[-1]))
    if getattr(config, "problem_type", None) is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    if config.problem_type == "regression":
        if num_labels == 1:
            return F.mse_loss(logits.squeeze(), labels.squeeze())
        return F.mse_loss(logits, labels)
    if config.problem_type == "single_label_classification":
        return F.cross_entropy(logits.view(-1, num_labels), labels.view(-1))
    if config.problem_type == "multi_label_classification":
        return F.binary_cross_entropy_with_logits(logits, labels)
    raise ValueError(f"Unsupported problem_type={config.problem_type!r}")


def fedrod_head_forward(
    model,
    *,
    head,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,
):
    kwargs.pop("num_items_in_batch", None)
    if labels is None:
        labels = kwargs.pop("label", None)
    if labels is None:
        labels = kwargs.pop("label_ids", None)

    sequence_model = _get_sequence_classifier_model(model)
    backbone = _get_fedrod_backbone(sequence_model)
    return_dict = return_dict if return_dict is not None else getattr(sequence_model.config, "use_return_dict", True)

    backbone_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
        "head_mask": head_mask,
        "inputs_embeds": inputs_embeds,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "return_dict": True,
    }
    backbone_kwargs = {key: value for key, value in backbone_kwargs.items() if value is not None}
    outputs = backbone(**backbone_kwargs)
    sequence_output = outputs[0]
    logits = head(sequence_output)
    loss = _sequence_classification_loss(sequence_model.config, logits, labels)

    if not return_dict:
        output = (logits,) + tuple(outputs[2:])
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def fedrod_forward(model, mode, home_cluster_idx=None, **inputs):
    if mode not in {"cluster", "universal", "mixed"}:
        raise ValueError(f"mode must be 'cluster', 'universal', or 'mixed', got {mode!r}")

    sequence_model = _get_sequence_classifier_model(model)
    if mode == "cluster":
        with uemd_forward_mode(model, "cluster_only"):
            return fedrod_head_forward(model, head=sequence_model.classifier, **inputs)
    if mode == "universal":
        with uemd_forward_mode(model, "universal_only"):
            return fedrod_head_forward(model, head=model.cls_universal, **inputs)

    labels = inputs.get("labels", inputs.get("label", inputs.get("label_ids")))
    cluster_outputs = fedrod_forward(model, "cluster", home_cluster_idx=home_cluster_idx, **inputs)
    universal_outputs = fedrod_forward(model, "universal", home_cluster_idx=home_cluster_idx, **inputs)
    logits = fedrod_mix_logits(model, cluster_outputs.logits, universal_outputs.logits, home_cluster_idx)
    loss = _sequence_classification_loss(sequence_model.config, logits, labels)
    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=cluster_outputs.hidden_states,
        attentions=cluster_outputs.attentions,
    )


class FedRoDTrainer(Trainer):
    def __init__(
        self,
        *args,
        fedrod_universal_coeff=1.0,
        fedrod_alpha_coeff=0.0,
        home_cluster_idx=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fedrod_universal_coeff = float(fedrod_universal_coeff)
        self.fedrod_alpha_coeff = float(fedrod_alpha_coeff)
        self.home_cluster_idx = home_cluster_idx

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cluster_outputs = fedrod_forward(
            model,
            "cluster",
            home_cluster_idx=self.home_cluster_idx,
            **inputs,
        )
        universal_outputs = fedrod_forward(
            model,
            "universal",
            home_cluster_idx=self.home_cluster_idx,
            **inputs,
        )

        loss = cluster_outputs.loss + self.fedrod_universal_coeff * universal_outputs.loss
        mixed_logits = fedrod_mix_logits(
            model,
            cluster_outputs.logits,
            universal_outputs.logits,
            self.home_cluster_idx,
        )
        labels = inputs.get("labels", inputs.get("label", inputs.get("label_ids")))
        mixed_loss = _sequence_classification_loss(model.config, mixed_logits, labels)
        if self.fedrod_alpha_coeff > 0:
            loss = loss + self.fedrod_alpha_coeff * mixed_loss

        outputs = SequenceClassifierOutput(
            loss=loss,
            logits=mixed_logits,
            hidden_states=cluster_outputs.hidden_states,
            attentions=cluster_outputs.attentions,
        )
        return (loss, outputs) if return_outputs else loss


class FedRoDEvalWrapper(nn.Module):
    def __init__(self, model, home_cluster_idx):
        super().__init__()
        self.model = model
        self.home_cluster_idx = home_cluster_idx
        self.config = model.config

    def forward(self, **inputs):
        return fedrod_forward(
            self.model,
            "mixed",
            home_cluster_idx=self.home_cluster_idx,
            **inputs,
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class Client:
    def __init__(
        self,
        client_id,
        task_name,
        tokenizer,
        model_name,
        num_clients,
        rank=8,
        lora_n=4,
        adaptive=False,
        cache_path="./output",
        idx=None,
        universal_idx=None,
        additive_residual=False,
        shared_lora_a=False,
        uemd_logit_coeff=0.0,
        visa_coeff=0.0,
        fedrod_dual_head=False,
        fedrod_alpha_init=2.0,
        fedrod_universal_coeff=1.0,
        fedrod_alpha_coeff=0.0,
        rdrop_kl_coeff=0.0,
        rdrop_direction="symmetric",
        rdrop_stopgrad="target",
    ):
        self.client_id = client_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num_clients = num_clients
        self.rank = rank
        self.lora_n = lora_n
        self.adaptive = adaptive
        self.cache_path = cache_path
        self.local_model = None
        self.current_params = None
        self.datasets = None
        self.num_labels = None
        self.idx = idx
        self.universal_idx = universal_idx
        self.additive_residual = additive_residual
        self.shared_lora_a = shared_lora_a
        self.uemd_logit_coeff = float(uemd_logit_coeff)
        self.visa_coeff = float(visa_coeff)
        self.fedrod_dual_head = bool(fedrod_dual_head)
        self.fedrod_alpha_init = float(fedrod_alpha_init)
        self.fedrod_universal_coeff = float(fedrod_universal_coeff)
        self.fedrod_alpha_coeff = float(fedrod_alpha_coeff)
        self.rdrop_kl_coeff = float(rdrop_kl_coeff)
        self.rdrop_direction = rdrop_direction
        self.rdrop_stopgrad = rdrop_stopgrad
        self._routing_hook_handles = []
        
    def set_dataset(self, dataset, num_labels):
        self.datasets = dataset
        self.num_labels = num_labels

    def set_adaptive(self, adaptive: bool):
        self.adaptive = adaptive

    def _clear_routing_hooks(self):
        for handle in self._routing_hook_handles:
            handle.remove()
        self._routing_hook_handles = []

    def _register_hard_routing_hooks(self):
        self._clear_routing_hooks()

        if self.idx is None or self.lora_n <= 1:
            return

        def force_assigned_expert(_module, _inputs, output):
            forced_logits = torch.full_like(output, torch.finfo(output.dtype).min / 2)
            forced_logits[..., self.idx] = 0.0
            return forced_logits

        for name, module in self.local_model.named_modules():
            if name.endswith("lora_route"):
                self._routing_hook_handles.append(module.register_forward_hook(force_assigned_expert))

    def _collapse_expert_probs(self, route_weight: torch.Tensor) -> torch.Tensor:
        return _collapse_expert_probs(route_weight, self.lora_n, self.idx)

    def _eval_forward_context(self):
        if self.local_model is None:
            return nullcontext()
        if self.visa_coeff > 0:
            return uemd_forward_mode(self.local_model, "home_cluster_only")
        if self.fedrod_dual_head:
            # FedRoD's mixed forward returns SequenceClassifierOutput from a wrapper
            # that confuses HF Trainer's eval-loss/metrics extraction (loses eval_accuracy).
            # Use cluster_only mode for eval — apples-to-apples with Stage 1 baseline,
            # tests whether FedRoD's gradient-isolated training improves cluster experts.
            return uemd_forward_mode(self.local_model, "cluster_only")
        return nullcontext()
    
    def load_model(self):
        if self.local_model is None:
            self.local_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_path,
                num_labels=self.num_labels,
            )
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=["query", "value"],
                inference_mode=False,
                r=self.rank,
                lora_alpha=16,
                lora_dropout=0.05,
                lora_nums=self.lora_n,
                adaptive=self.adaptive,
                idx=self.idx,
                k=self.lora_n,
                universal_idx=self.universal_idx if self.additive_residual else None,
                shared_lora_a=self.shared_lora_a,
                bias="none"
            )
            
            self.local_model = get_peft_model(self.local_model, peft_config)
            if self.fedrod_dual_head:
                ensure_fedrod_dual_head(
                    self.local_model,
                    lora_n=self.lora_n,
                    universal_idx=self.universal_idx,
                    alpha_init=self.fedrod_alpha_init,
                )

            if not self.adaptive:
                self._register_hard_routing_hooks()
            
            if self.current_params is not None:
                self.load_params(self.current_params)
    
    def unload_model(self):
        if self.local_model is not None:
            self._clear_routing_hooks()
            self.current_params = self.get_lora_params()['params']
            del self.local_model
            self.local_model = None
            torch.cuda.empty_cache()
            gc.collect()
        
    def get_lora_params(self):
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }
        for name, param in self.local_model.named_parameters():
            if (
                'lora_A' in name
                or 'lora_B' in name
                or 'lora_route' in name
                or 'classifier' in name
                or 'cls_universal' in name
                or name.endswith('alpha_logit')
            ):
                lora_params['params'][name] = param.data.clone()
        return lora_params
    
    def get_lora_params_and_save_by_module(self, round_id, personal_dir):
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }

        target_modules = ["query", "value"]
        lora_A_dict = {module: [] for module in target_modules}
        lora_B_dict = {module: [] for module in target_modules}

        for name, param in self.local_model.named_parameters():
            if (
                'lora_A' in name
                or 'lora_B' in name
                or 'lora_route' in name
                or 'classifier' in name
                or 'cls_universal' in name
                or name.endswith('alpha_logit')
            ):
                lora_params['params'][name] = param.data.clone()

            for module in target_modules:
                if module in name:
                    if 'lora_A0' in name:
                        lora_A_dict[module].append(param.data.clone().cpu().numpy())
                    elif 'lora_B0' in name:
                        lora_B_dict[module].append(param.data.clone().cpu().numpy())

        param_dir = os.path.join(personal_dir, "lora_params")
        os.makedirs(param_dir, exist_ok=True)

        for module in target_modules:
            if lora_A_dict[module]:  
                np.save(os.path.join(param_dir, f'{module}_lora_A_client_{self.client_id}_{round_id}.npy'), np.array(lora_A_dict[module]))
            if lora_B_dict[module]:  
                np.save(os.path.join(param_dir, f'{module}_lora_B_client_{self.client_id}_{round_id}.npy'), np.array(lora_B_dict[module]))

        return lora_params
    
    def load_params(self, params_or_path):
        if isinstance(params_or_path, dict):
            params_to_load = params_or_path['params'] if 'params' in params_or_path else params_or_path
            current_state = self.local_model.state_dict()
            filtered_params = {}
            skipped_params = []

            for name, value in params_to_load.items():
                if name not in current_state:
                    skipped_params.append(f"{name} (missing)")
                    continue
                if current_state[name].shape != value.shape:
                    skipped_params.append(
                        f"{name} (checkpoint {tuple(value.shape)} != model {tuple(current_state[name].shape)})"
                    )
                    continue
                filtered_params[name] = value

            self.local_model.load_state_dict(filtered_params, strict=False)

            if skipped_params:
                print(
                    f"Client {self.client_id}: skipped {len(skipped_params)} incompatible params "
                    f"while loading state"
                )
        else:
            self.local_model.load_adapter(params_or_path, adapter_name="default")
            

    def local_training(
        self,
        lr=2e-4,
        epochs=1,
        batch_size=32,
        gradient_accumulation_steps=1,
        lora_client_map=None,
        in_universal_warmup=False,
        load_balance_coeff=0.0,
        uemd_coeff=0.0,
        uemd_logit_coeff=None,
        visa_coeff=0.0,
        soft_membership_for_client=None,
        shared_lora_a=False,
        fedrod_dual_head=None,
        fedrod_universal_coeff=None,
        fedrod_alpha_coeff=None,
        affinity_mode="off",
    ):
        self.local_model.train()
        if uemd_logit_coeff is None:
            uemd_logit_coeff = self.uemd_logit_coeff
        visa_coeff = float(visa_coeff)
        self.visa_coeff = visa_coeff
        if fedrod_dual_head is None:
            fedrod_dual_head = self.fedrod_dual_head
        fedrod_dual_head = bool(fedrod_dual_head)
        if fedrod_universal_coeff is None:
            fedrod_universal_coeff = self.fedrod_universal_coeff
        if fedrod_alpha_coeff is None:
            fedrod_alpha_coeff = self.fedrod_alpha_coeff
        fedrod_universal_coeff = float(fedrod_universal_coeff)
        fedrod_alpha_coeff = float(fedrod_alpha_coeff)

        if lora_client_map is None:
            raise ValueError("lora_client_map is required for local_training after warmup")

        client_lora_group = None
        for lora_idx, client_indices in lora_client_map.items():
            if self.client_id in client_indices:
                client_lora_group = int(lora_idx)
                break

        if client_lora_group is None:
            print(f"Client {self.client_id} is a dummy client or not found in lora_client_map")
            print(f"Training all LoRA modules for client {self.client_id}")

            for name, param in self.local_model.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    param.requires_grad = True
                elif 'lora_route' in name:
                    param.requires_grad = self.adaptive
                elif 'classifier' in name:
                    param.requires_grad = True
                elif 'cls_universal' in name or name.endswith('alpha_logit'):
                    param.requires_grad = fedrod_dual_head
                else:
                    param.requires_grad = False
        else:
            print(f"Client {self.client_id} belongs to LoRA group {client_lora_group}")
            trainable_experts = _build_trainable_experts(
                client_lora_group,
                self.universal_idx,
                soft_membership_for_client=soft_membership_for_client,
            )
            if visa_coeff > 0 and not in_universal_warmup and self.universal_idx is not None:
                trainable_experts.discard(int(self.universal_idx))
            for name, param in self.local_model.named_parameters():
                if in_universal_warmup and self.universal_idx is not None:
                    if 'lora_A' in name:
                        param.requires_grad = True
                    elif 'lora_B' in name:
                        param.requires_grad = _matches_lora_expert(name, self.universal_idx)
                    elif 'lora_route' in name:
                        param.requires_grad = self.adaptive
                    elif 'classifier' in name:
                        param.requires_grad = True
                    elif 'cls_universal' in name or name.endswith('alpha_logit'):
                        param.requires_grad = fedrod_dual_head
                    else:
                        param.requires_grad = False
                elif 'lora_route' in name:
                    param.requires_grad = self.adaptive
                elif 'classifier' in name:
                    param.requires_grad = True
                elif 'cls_universal' in name:
                    param.requires_grad = fedrod_dual_head
                elif name.endswith('alpha_logit'):
                    param.requires_grad = fedrod_dual_head
                elif shared_lora_a and 'lora_A' in name:
                    # HydraFed-LEASE: all lora_A_k trainable on every client
                    # so that A receives gradient signal from all tasks; aggregation
                    # then pools A_k across all clients (server-side).
                    param.requires_grad = True
                elif any(_matches_lora_expert(name, expert_idx) for expert_idx in trainable_experts):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        trainable_params = [p for p in self.local_model.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {len(trainable_params)}")
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")

        training_args = TrainingArguments(
            output_dir=f"{self.cache_path}/{self.rank}_{self.lora_n}_proposed/client_{self.client_id}_checkpoints",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=5,
            optim="adamw_torch",
            weight_decay=0.05,
            eval_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            remove_unused_columns=False,
            gradient_checkpointing=False
        )

        trainer_cls = Trainer
        trainer_kwargs = {}
        enabled_aux = sum(
            bool(enabled)
            for enabled in (
                float(uemd_coeff) > 0,
                float(uemd_logit_coeff) > 0,
                visa_coeff > 0,
                fedrod_dual_head,
            )
        )
        if enabled_aux > 1:
            raise ValueError("--uemd_coeff, --uemd_logit_coeff, --visa_coeff, and --fedrod_dual_head are mutually exclusive")
        if fedrod_dual_head and not in_universal_warmup:
            if self.universal_idx is None or not self.additive_residual:
                raise ValueError("--fedrod_dual_head requires --universal_expert with --additive_residual")
            ensure_fedrod_dual_head(
                self.local_model,
                lora_n=self.lora_n,
                universal_idx=self.universal_idx,
                alpha_init=self.fedrod_alpha_init,
            )
            trainer_cls = FedRoDTrainer
            trainer_kwargs = {
                "fedrod_universal_coeff": fedrod_universal_coeff,
                "fedrod_alpha_coeff": fedrod_alpha_coeff,
                "home_cluster_idx": client_lora_group,
            }
        elif visa_coeff > 0 and not in_universal_warmup:
            if self.universal_idx is None or not self.additive_residual:
                raise ValueError("--visa_coeff requires --universal_expert with --additive_residual")
            if not soft_membership_for_client:
                raise ValueError("--visa_coeff requires --soft_membership task_family")
            trainer_cls = WriteOnlySoftTrainer
            trainer_kwargs = {
                "visa_coeff": visa_coeff,
                "soft_membership_for_client": soft_membership_for_client,
                "lora_n": self.lora_n,
                "universal_idx": self.universal_idx,
                "home_idx": client_lora_group,
                "affinity_mode": affinity_mode,
                "rdrop_kl_coeff": self.rdrop_kl_coeff,
                "rdrop_direction": self.rdrop_direction,
                "rdrop_stopgrad": self.rdrop_stopgrad,
            }
        elif uemd_logit_coeff > 0:
            if self.universal_idx is None or not self.additive_residual:
                raise ValueError("--uemd_logit_coeff requires --universal_expert with --additive_residual")
            if load_balance_coeff > 0:
                raise ValueError("--uemd_logit_coeff and load_balance_coeff cannot both be positive")
            trainer_cls = LogitUEMDTrainer
            trainer_kwargs = {
                "uemd_logit_coeff": uemd_logit_coeff,
            }
        elif uemd_coeff > 0:
            if self.universal_idx is None or not self.additive_residual:
                raise ValueError("--uemd_coeff requires --universal_expert with --additive_residual")
            if load_balance_coeff > 0:
                raise ValueError("--uemd_coeff and load_balance_coeff cannot both be positive")
            trainer_cls = UEMDTrainer
            trainer_kwargs = {
                "uemd_coeff": uemd_coeff,
                "lora_n": self.lora_n,
                "universal_idx": self.universal_idx,
            }
        elif (
            self.universal_idx is not None
            and not self.additive_residual
            and not in_universal_warmup
            and self.adaptive
            and load_balance_coeff > 0
        ):
            trainer_cls = LoadBalancedTrainer
            trainer_kwargs = {
                "load_balance_coeff": load_balance_coeff,
                "lora_n": self.lora_n,
                "assigned_idx": self.idx,
            }

        trainer = trainer_cls(
            model=self.local_model,
            args=training_args,
            train_dataset=self.datasets["train"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            **trainer_kwargs,
        )

        try:
            trainer.train()
        finally:
            if isinstance(trainer, (LoadBalancedTrainer, UEMDTrainer)):
                trainer.close()

    def get_routing_stats(self, dataset=None, batch_size=8, max_batches=1):
        dataset = dataset or self.datasets["validation"]
        if dataset is None or self.idx is None:
            return {}

        loaded_here = False
        if self.local_model is None:
            self.load_model()
            loaded_here = True

        route_outputs = {}
        capture_handles = []

        def capture_route_logits(name):
            def hook(_module, _inputs, output):
                route_outputs.setdefault(name, []).append(output.detach().float().cpu())
            return hook

        for name, module in self.local_model.named_modules():
            if name.endswith("lora_route"):
                capture_handles.append(module.register_forward_hook(capture_route_logits(name)))

        try:
            eval_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=DataCollatorWithPadding(self.tokenizer),
            )

            device = next(self.local_model.parameters()).device
            self.local_model.eval()

            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    if batch_idx >= max_batches:
                        break
                    batch = {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    with self._eval_forward_context():
                        self.local_model(**batch)
        finally:
            for handle in capture_handles:
                handle.remove()
            if loaded_here:
                self.unload_model()

        if not route_outputs:
            return {}

        assigned_mass = []
        cross_mass = []
        universal_mass = []
        cluster_max_weight = []
        per_expert_mass = []
        entropy = []

        for outputs in route_outputs.values():
            flat_logits = torch.cat([tensor.reshape(-1, tensor.shape[-1]) for tensor in outputs], dim=0)
            route_weight = torch.softmax(flat_logits, dim=-1)
            if self.additive_residual and self.universal_idx is not None:
                expert_weight = _collapse_additive_residual_probs(
                    route_weight,
                    self.lora_n,
                    self.idx,
                    self.universal_idx,
                )
            else:
                expert_weight = self._collapse_expert_probs(route_weight)

            assigned = expert_weight[:, self.idx]
            cross = expert_weight.sum(dim=-1) - assigned
            module_entropy = -(expert_weight * expert_weight.clamp_min(1e-12).log()).sum(dim=-1)

            assigned_mass.append(assigned.mean().item())
            cross_mass.append(cross.mean().item())
            entropy.append(module_entropy.mean().item())

            if self.universal_idx is not None and self.universal_idx < expert_weight.shape[-1]:
                universal_mass.append(expert_weight[:, self.universal_idx].mean().item())
                cluster_mask = torch.ones(expert_weight.shape[-1], dtype=torch.bool)
                cluster_mask[self.universal_idx] = False
                cluster_weight = expert_weight[:, cluster_mask]
            else:
                cluster_weight = expert_weight

            if cluster_weight.numel() > 0:
                cluster_max_weight.append(cluster_weight.max(dim=-1).values.mean().item())

            per_expert_mass.append(expert_weight.mean(dim=0).tolist())

        stats = {
            "assigned_mass": float(np.mean(assigned_mass)),
            "cross_mass": float(np.mean(cross_mass)),
            "routing_entropy": float(np.mean(entropy)),
            "num_routing_modules": len(route_outputs),
        }

        if universal_mass:
            stats["universal_mass"] = float(np.mean(universal_mass))

        if cluster_max_weight:
            stats["cluster_max_weight"] = float(np.mean(cluster_max_weight))

        if per_expert_mass:
            avg_per_expert = np.mean(per_expert_mass, axis=0).tolist()
            stats["per_expert_mass"] = [round(v, 6) for v in avg_per_expert]

        return stats
        
    def evaluate_on_dataset(self, dataset, num_labels=None, output_file=None, dataset_name="validation"):
        if num_labels is not None:
            self.num_labels = num_labels

        loaded_here = False
        if self.local_model is None:
            self.load_model()
            loaded_here = True

        if num_labels is not None and self.local_model.config.num_labels != num_labels:
            raise ValueError(
                f"Client {self.client_id} model head has num_labels={self.local_model.config.num_labels}, "
                f"expected {num_labels}"
            )

        self.local_model.eval()

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

        eval_args = TrainingArguments(
            output_dir=f"{self.cache_path}/temp_eval_output",
            per_device_eval_batch_size=256,
            fp16=True,
            report_to="none",
        )
        # FedRoD eval uses cluster_only mode via _eval_forward_context, NOT the wrapper —
        # wrapper's mixed-forward broke Trainer.evaluate's logits/loss extraction.
        eval_model = self.local_model
        trainer = Trainer(
            model=eval_model,
            args=eval_args,
            eval_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=compute_metrics 
        )

        with self._eval_forward_context():
            metrics = trainer.evaluate()

        print(f"Evaluation metrics for client {self.client_id} on {dataset_name} dataset:")
        print(metrics)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'client_id': self.client_id,
                    'task': self.task_name,
                    'dataset_type': dataset_name,
                    'metrics': metrics
                }, f, indent=2)
            print(f'The output file is stored at {output_file}')

        if loaded_here:
            self.unload_model()

        return metrics

    def evaluate_model(self, output_file=None):
        return self.evaluate_on_dataset(
            self.datasets["validation"],
            num_labels=self.num_labels,
            output_file=output_file,
            dataset_name="validation",
        )



class WarmupClient(Client):
    def __init__(self, client_id, task_name, tokenizer, model_name, num_clients, rank=8, cache_path="./output"):
        
        super().__init__(client_id, task_name, tokenizer, model_name, num_clients, rank, lora_n=1, adaptive=False, cache_path=cache_path)
        
    def load_model(self):
        
        if self.local_model is None:
            self.local_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_path,
                num_labels=self.num_labels,
            )
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=["query", "value"],
                inference_mode=False,
                r=self.rank,
                lora_alpha=16,
                lora_dropout=0.05,
                lora_nums=1,
                adaptive=False, 
                idx=0,  
                k=1, 
                bias="none"
            )
            
            self.local_model = get_peft_model(self.local_model, peft_config)
            
            if self.current_params is not None:
                self.load_params(self.current_params)

    def local_training(self, lr=2e-4, epochs=1, batch_size=32, gradient_accumulation_steps=1):
        self.local_model.train()

        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_params = [p for p in self.local_model.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {len(trainable_params)}")
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found!")

        training_args = TrainingArguments(
            output_dir=f"{self.cache_path}/warmup/client_{self.client_id}_checkpoints",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=5,
            optim="adamw_torch",
            weight_decay=0.05,
            eval_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            remove_unused_columns=False,
            gradient_checkpointing=False
        )

        trainer = Trainer(
            model=self.local_model,
            args=training_args,
            train_dataset=self.datasets["train"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )

        trainer.train()

    def get_lora_params(self):
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }
        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'classifier' in name:
                lora_params['params'][name] = param.data.clone()
        return lora_params
    
    def get_lora_params_and_save_by_module(self, round_id, personal_dir):
        lora_params = {
            'client_id': self.client_id,
            'params': {}
        }

        target_modules = ["query", "value"]
        lora_A_dict = {module: [] for module in target_modules}
        lora_B_dict = {module: [] for module in target_modules}

        for name, param in self.local_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'lora_route' in name or 'classifier' in name:
                lora_params['params'][name] = param.data.clone()

            for module in target_modules:
                if module in name:
                    if 'lora_A0' in name:
                        lora_A_dict[module].append(param.data.clone().cpu().numpy())
                    elif 'lora_B0' in name:
                        lora_B_dict[module].append(param.data.clone().cpu().numpy())

        param_dir = os.path.join(personal_dir, "lora_params")
        print(param_dir)
        os.makedirs(param_dir, exist_ok=True)

        for module in target_modules:
            if lora_A_dict[module]:  
                np.save(os.path.join(param_dir, f'{module}_lora_A_client_{self.client_id}_{round_id}.npy'), np.array(lora_A_dict[module]))
            if lora_B_dict[module]: 
                np.save(os.path.join(param_dir, f'{module}_lora_B_client_{self.client_id}_{round_id}.npy'), np.array(lora_B_dict[module]))

        return lora_params


def _matches_lora_expert(param_name: str, expert_idx: int) -> bool:
    return re.search(rf"lora_[AB]{expert_idx}(?!\d)", param_name) is not None

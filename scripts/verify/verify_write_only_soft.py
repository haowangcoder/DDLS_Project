#!/usr/bin/env python
"""Smoke checks for write-only soft-membership training.

Default checks are CPU/GPU friendly and use a tiny randomly initialized
RoBERTa model. The large-model memory check is opt-in via --memory-test.
"""

import argparse
import os
import re
import sys
import tempfile

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from client import WriteOnlySoftTrainer  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from peft.tuners.lora import uemd_forward_mode  # noqa: E402


HOME_IDX = 0
UNIVERSAL_IDX = 3
SOFT_MEMBERSHIP = {0: 0.80, 1: 0.15, 2: 0.05}


def _make_roberta_config(
    *,
    vocab_size=97,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=128,
    max_position_embeddings=128,
    num_labels=2,
):
    return RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
        num_labels=num_labels,
    )


def _build_lora_model(config, device, dtype=torch.float32):
    base_model = RobertaForSequenceClassification(config)
    if dtype != torch.float32:
        base_model = base_model.to(dtype=dtype)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["query", "value"],
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_nums=4,
        adaptive=True,
        idx=HOME_IDX,
        k=4,
        universal_idx=UNIVERSAL_IDX,
        bias="none",
    )
    model = get_peft_model(base_model, peft_config).to(device)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.normal_(mean=0.0, std=0.05)
            elif "lora_B" in name:
                param.normal_(mean=0.0, std=0.05)
            elif "lora_route" in name:
                param.normal_(mean=0.0, std=0.05)
    return model


def _make_batch(config, batch_size, seq_len, device):
    input_ids = torch.randint(4, config.vocab_size, (batch_size, seq_len), device=device)
    input_ids[:, 0] = 0
    input_ids[:, -1] = 2
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.randint(0, config.num_labels, (batch_size,), device=device),
    }


def _training_args():
    tmpdir = tempfile.TemporaryDirectory()
    args = TrainingArguments(output_dir=tmpdir.name, report_to="none")
    return tmpdir, args


def _make_trainer(model, args, visa_coeff):
    return WriteOnlySoftTrainer(
        model=model,
        args=args,
        visa_coeff=visa_coeff,
        soft_membership_for_client=SOFT_MEMBERSHIP,
        lora_n=4,
        universal_idx=UNIVERSAL_IDX,
        home_idx=HOME_IDX,
    )


def _extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs[1] if len(outputs) > 1 else outputs[0]


def _logits(model, batch):
    no_label_batch = {key: value for key, value in batch.items() if key != "labels"}
    return model(**no_label_batch).logits


def _assert_different(name_a, tensor_a, name_b, tensor_b):
    if torch.allclose(tensor_a, tensor_b):
        raise AssertionError(f"{name_a} and {name_b} logits should differ")


def _expert_pattern(expert_idx):
    return re.compile(rf"lora_[AB]{expert_idx}(?!\d)")


def _zero_grads(model):
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None


def _expert_grad_sum(model, expert_idx):
    pattern = _expert_pattern(expert_idx)
    total = 0.0
    for name, param in model.named_parameters():
        if pattern.search(name) and param.grad is not None:
            total += param.grad.detach().abs().sum().item()
    return total


def _route_grad_sum(model):
    total = 0.0
    for name, param in model.named_parameters():
        if "lora_route" in name and param.grad is not None:
            total += param.grad.detach().abs().sum().item()
    return total


def _assert_nonzero(label, value, threshold=1e-9):
    if value <= threshold:
        raise AssertionError(f"{label} expected nonzero gradient, got {value:.3e}")


def _assert_zero(label, value, threshold=1e-9):
    if value > threshold:
        raise AssertionError(f"{label} expected zero gradient, got {value:.3e}")


def verify_zero_coeff_home_path(model, batch):
    model.eval()
    tmpdir, args = _training_args()
    try:
        trainer = _make_trainer(model, args, visa_coeff=0.0)
        with torch.no_grad():
            with uemd_forward_mode(model, "home_cluster_only"):
                ref_outputs = model(**batch)
        loss, outputs = trainer.compute_loss(
            model,
            {key: value.clone() for key, value in batch.items()},
            return_outputs=True,
        )
    finally:
        tmpdir.cleanup()

    if not torch.equal(ref_outputs.loss.detach(), loss.detach()):
        raise AssertionError("visa_coeff=0 changed home-only CE loss")
    if not torch.equal(ref_outputs.logits.detach(), _extract_logits(outputs).detach()):
        raise AssertionError("visa_coeff=0 changed home-only logits")
    print("PASS CU8.a: visa_coeff=0 is bit-identical to explicit home_cluster_only CE")


def verify_pass_logits_and_grads(model, batch):
    model.train()
    tmpdir, args = _training_args()
    try:
        trainer = _make_trainer(model, args, visa_coeff=0.2)
        trainer._ensure_visa_weights(model)

        with torch.no_grad():
            with uemd_forward_mode(model, "home_cluster_only"):
                home_logits = _logits(model, batch)
            with uemd_forward_mode(model, "non_home_visa"):
                visa_logits = _logits(model, batch)
        _assert_different("home_cluster_only", home_logits, "non_home_visa", visa_logits)

        _zero_grads(model)
        with uemd_forward_mode(model, "home_cluster_only"):
            home_loss = model(**batch).loss
        home_loss.backward()
        home_grad = _expert_grad_sum(model, HOME_IDX)
        foreign_grad = _expert_grad_sum(model, 1) + _expert_grad_sum(model, 2)
        universal_grad = _expert_grad_sum(model, UNIVERSAL_IDX)
        route_grad = _route_grad_sum(model)
        _assert_nonzero("home pass home expert", home_grad)
        _assert_zero("home pass foreign experts", foreign_grad)
        _assert_zero("home pass universal expert", universal_grad)
        _assert_zero("home pass router", route_grad)

        _zero_grads(model)
        with uemd_forward_mode(model, "non_home_visa"):
            visa_loss = model(**batch).loss
        visa_loss.backward()
        home_grad = _expert_grad_sum(model, HOME_IDX)
        foreign_grad_1 = _expert_grad_sum(model, 1)
        foreign_grad_2 = _expert_grad_sum(model, 2)
        universal_grad = _expert_grad_sum(model, UNIVERSAL_IDX)
        route_grad = _route_grad_sum(model)
        _assert_zero("visa pass home expert", home_grad)
        _assert_nonzero("visa pass foreign expert 1", foreign_grad_1)
        _assert_nonzero("visa pass foreign expert 2", foreign_grad_2)
        _assert_zero("visa pass universal expert", universal_grad)
        _assert_zero("visa pass router", route_grad)

        _zero_grads(model)
        total_loss, outputs = trainer.compute_loss(
            model,
            {key: value.clone() for key, value in batch.items()},
            return_outputs=True,
        )
        total_loss.backward()
        if total_loss.ndim != 0 or _extract_logits(outputs).shape != home_logits.shape:
            raise AssertionError("WriteOnlySoftTrainer total loss/output shape check failed")
        _assert_nonzero("combined home expert", _expert_grad_sum(model, HOME_IDX))
        _assert_nonzero(
            "combined foreign experts",
            _expert_grad_sum(model, 1) + _expert_grad_sum(model, 2),
        )
        _assert_zero("combined universal expert", _expert_grad_sum(model, UNIVERSAL_IDX))
        _assert_zero("combined router", _route_grad_sum(model))
    finally:
        tmpdir.cleanup()

    print("PASS CU8.b: home/visa logits differ and gradients are routed write-only")


def verify_eval_home_mode_after_training(model, batch):
    model.train()
    tmpdir, args = _training_args()
    try:
        trainer = _make_trainer(model, args, visa_coeff=0.2)
        optimizer = torch.optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=0.05,
        )

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = trainer.compute_loss(
                model,
                {key: value.clone() for key, value in batch.items()},
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            full_logits = _logits(model, batch)
            with uemd_forward_mode(model, "home_cluster_only"):
                home_eval_logits = _logits(model, batch)
        _assert_different("full eval", full_logits, "home_cluster_only eval", home_eval_logits)
    finally:
        tmpdir.cleanup()

    print("PASS CU8.c: eval home_cluster_only logits differ from full-mode routing after write-only steps")


def run_large_memory_test(args):
    if not args.memory_test:
        print("SKIP memory test: pass --memory-test on an H200 node to check RoBERTa-large bs=64")
        return
    if not torch.cuda.is_available():
        print("SKIP memory test: CUDA is unavailable")
        return

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    config = _make_roberta_config(
        vocab_size=50265,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=max(args.memory_seq_len + 2, 514),
        num_labels=2,
    )
    model = _build_lora_model(config, device=device, dtype=torch.float16)
    model.train()
    batch = _make_batch(config, args.memory_batch_size, args.memory_seq_len, device)
    tmpdir, training_args = _training_args()

    try:
        trainer = _make_trainer(model, training_args, visa_coeff=0.2)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss, _outputs = trainer.compute_loss(model, batch, return_outputs=True)
            _ = loss.detach()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise AssertionError(
                f"RoBERTa-large 2-pass memory test OOM at bs={args.memory_batch_size}, "
                f"seq_len={args.memory_seq_len}"
            ) from exc
        raise
    finally:
        tmpdir.cleanup()
        del model
        torch.cuda.empty_cache()

    print(
        f"PASS CU8.d: RoBERTa-large 2-pass forward bs={args.memory_batch_size}, "
        f"seq_len={args.memory_seq_len}, peak={peak_gb:.2f} GiB"
    )


def main():
    parser = argparse.ArgumentParser(description="Verify write-only soft-membership training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--memory-test", action="store_true")
    parser.add_argument("--memory-batch-size", type=int, default=64)
    parser.add_argument("--memory-seq-len", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device(args.device)
    config = _make_roberta_config(max_position_embeddings=max(args.seq_len + 2, 32))
    model = _build_lora_model(config, device=device)
    batch = _make_batch(config, args.batch_size, args.seq_len, device)

    verify_zero_coeff_home_path(model, batch)
    verify_pass_logits_and_grads(model, batch)
    eval_model = _build_lora_model(config, device=device)
    verify_eval_home_mode_after_training(eval_model, batch)
    run_large_memory_test(args)
    print("\nAll write-only-soft checks PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

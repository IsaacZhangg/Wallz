"""Policy-value optimization over self-play replay examples."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from wallzero.encoding import encode_state, mirror_policy, mirror_state
from wallzero.network import PolicyValueNet
from wallzero.replay import TrainingExample


@dataclass(frozen=True, slots=True)
class TrainConfig:
    steps: int = 1_000
    batch_size: int = 512
    learning_rate: float = 2e-3
    minimum_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    warmup_fraction: float = 0.05
    mirror_probability: float = 0.5
    seed: int = 1

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("training steps and batch size must be positive")
        if not 0.0 <= self.mirror_probability <= 1.0:
            raise ValueError("mirror_probability must be between zero and one")


@dataclass(slots=True)
class TrainMetrics:
    steps: int
    samples: int
    policy_loss: float
    value_loss: float
    total_loss: float
    policy_entropy: float
    elapsed_seconds: float


def _learning_rate_multiplier(step: int, config: TrainConfig) -> float:
    warmup_steps = max(1, round(config.steps * config.warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, config.steps - warmup_steps - 1)
    minimum_ratio = config.minimum_learning_rate / config.learning_rate
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def train_candidate(
    model: PolicyValueNet,
    examples: list[TrainingExample],
    device: torch.device,
    config: TrainConfig,
) -> tuple[TrainMetrics, torch.optim.Optimizer]:
    if not examples:
        raise ValueError("cannot train without self-play examples")
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_multiplier(step, config),
    )
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_float16 = device.type == "cuda" and not use_bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_float16)
    autocast_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    policy_loss_total = 0.0
    value_loss_total = 0.0
    total_loss_total = 0.0
    entropy_total = 0.0
    started = time.monotonic()

    frequency = np.array([example.weight for example in examples], dtype=np.float64)
    uniform_weights = bool(np.all(frequency == frequency[0]))
    sampling_weights = None if uniform_weights else frequency / frequency.sum()
    distance_head = getattr(model, "distance_output", None) is not None

    for _ in range(config.steps):
        if sampling_weights is None:
            indices = rng.integers(0, len(examples), size=config.batch_size)
        else:
            indices = rng.choice(
                len(examples), size=config.batch_size, p=sampling_weights
            )
        states = []
        policies = []
        values = []
        policy_weights = []
        distances = []
        for index in indices:
            example = examples[int(index)]
            if rng.random() < config.mirror_probability:
                states.append(encode_state(mirror_state(example.state)))
                policies.append(mirror_policy(example.policy))
            else:
                states.append(encode_state(example.state))
                policies.append(example.policy)
            values.append(example.value)
            policy_weights.append(example.policy_weight)
            distances.append((example.own_distance, example.opp_distance))

        inputs = torch.from_numpy(np.stack(states)).to(device, non_blocking=True)
        target_policy = torch.from_numpy(np.stack(policies)).to(
            device, non_blocking=True
        )
        target_value = torch.tensor(values, dtype=torch.float32, device=device)
        policy_mask = torch.tensor(policy_weights, dtype=torch.float32, device=device)
        target_distance = torch.tensor(distances, dtype=torch.float32, device=device)

        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            policy_logits, predicted_value, predicted_distance = model.forward_train(
                inputs
            )
            log_policy = torch.log_softmax(policy_logits, dim=1)
            per_sample_policy = -(target_policy * log_policy).sum(dim=1)
            mask_total = policy_mask.sum().clamp(min=1.0)
            policy_loss = (per_sample_policy * policy_mask).sum() / mask_total
            value_loss = nn.functional.mse_loss(predicted_value, target_value)
            loss = policy_loss + value_loss
            if distance_head and predicted_distance is not None:
                valid = (target_distance >= 0.0).all(dim=1).float()
                distance_error = (
                    (predicted_distance - target_distance / 20.0) ** 2
                ).mean(dim=1)
                distance_loss = (distance_error * valid).sum() / valid.sum().clamp(
                    min=1.0
                )
                loss = loss + 0.1 * distance_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            probabilities = torch.softmax(policy_logits.float(), dim=1)
            entropy = (
                -(probabilities * torch.log_softmax(policy_logits.float(), dim=1))
                .sum(dim=1)
                .mean()
            )
        policy_loss_total += float(policy_loss.detach())
        value_loss_total += float(value_loss.detach())
        total_loss_total += float(loss.detach())
        entropy_total += float(entropy)

    elapsed = time.monotonic() - started
    model.eval()
    denominator = config.steps
    return (
        TrainMetrics(
            steps=config.steps,
            samples=config.steps * config.batch_size,
            policy_loss=policy_loss_total / denominator,
            value_loss=value_loss_total / denominator,
            total_loss=total_loss_total / denominator,
            policy_entropy=entropy_total / denominator,
            elapsed_seconds=elapsed,
        ),
        optimizer,
    )

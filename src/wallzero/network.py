"""Residual policy-value network, device selection, and checkpoint I/O."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from wallzero.constants import ACTION_SIZE, INPUT_PLANES
from wallzero.encoding import encode_state
from wallzero.game import State

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    input_planes: int = INPUT_PLANES
    channels: int = 128
    blocks: int = 10
    action_size: int = ACTION_SIZE
    value_hidden: int = 256
    squeeze_excite_ratio: int = 8


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: int) -> None:
        super().__init__()
        hidden = max(8, channels // ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        gates = self.pool(inputs)
        gates = torch.relu(self.fc1(gates))
        gates = torch.sigmoid(self.fc2(gates))
        return inputs * gates


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, squeeze_excite_ratio: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.squeeze_excite = SqueezeExcite(channels, squeeze_excite_ratio)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = torch.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        outputs = self.squeeze_excite(outputs)
        return torch.relu(outputs + residual)


class PolicyValueNet(nn.Module):
    """A configurable AlphaZero-style residual policy-value network."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        config = config or NetworkConfig()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.input_planes,
                config.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *(
                ResidualBlock(config.channels, config.squeeze_excite_ratio)
                for _ in range(config.blocks)
            )
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, config.action_size),
        )
        self.value_features = nn.Sequential(
            nn.Conv2d(config.channels, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 9 * 9, config.value_hidden),
            nn.ReLU(inplace=True),
        )
        self.value_output = nn.Linear(config.value_hidden, 1)
        self.apply(_initialize)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.tower(self.stem(inputs))
        policy_logits = self.policy_head(features)
        value = torch.tanh(self.value_output(self.value_features(features)))
        return policy_logits, value.squeeze(-1)


def _initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchEvaluator:
    """Batched, inference-only adapter used by MCTS."""

    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        *,
        amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.amp = amp and device.type == "cuda"
        self.model.eval()
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")

    def __call__(self, states: list[State]) -> tuple[FloatArray, FloatArray]:
        if not states:
            return (
                np.empty((0, ACTION_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return self.evaluate_planes(np.stack([encode_state(state) for state in states]))

    def evaluate_planes(self, planes: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate pre-encoded input planes, e.g. shipped from actor processes."""
        if len(planes) == 0:
            return (
                np.empty((0, ACTION_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        inputs = torch.from_numpy(planes).to(self.device, non_blocking=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.amp
            else nullcontext()
        )
        with torch.inference_mode(), context:
            logits, values = self.model(inputs)
        return (
            logits.float().cpu().numpy(),
            values.float().cpu().numpy(),
        )


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "network_config": asdict(model.config),
        "model_state": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PolicyValueNet, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format_version')}"
        )
    config = NetworkConfig(**payload["network_config"])
    model = PolicyValueNet(config)
    model.load_state_dict(payload["model_state"])
    return model, payload

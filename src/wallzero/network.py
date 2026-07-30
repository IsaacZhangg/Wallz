"""Residual policy-value network, device selection, and checkpoint I/O."""

from __future__ import annotations

import math
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
    distance_head: bool = False
    # KataGo-style trunk (audit round 7). "bnorm" is the original
    # BatchNorm+SE architecture and stays the default so existing checkpoints
    # reconstruct with identical state-dict keys. "fixscaleonenorm" replaces
    # per-layer BatchNorm with fixed variance-normalising scalars, keeps a
    # single BatchNorm at the trunk end, and attaches two independent head
    # sets: the BatchNorm path carries most of the loss and drives
    # optimisation, the BatchNorm-free path is what runs at inference.
    norm_kind: str = "bnorm"
    # 1-indexed trunk positions that use a global-pooling block. KataGo's
    # b10c128 pools at blocks 5 and 8 of 10.
    gpool_blocks: tuple[int, ...] = ()
    gpool_channels: int = 32
    # Loss share of the BatchNorm head path; the inference path gets the rest.
    bn_head_loss_weight: float = 0.8


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


class FixedScaleNorm(nn.Module):
    """Normalisation by a fixed constant instead of batch statistics.

    ``scale`` is chosen so that the idealised output variance is 1: treating
    each conv-activation pair as variance preserving and each block as adding
    variance 1 to the trunk, block ``i`` reads a trunk of variance ``i+1`` and
    so scales by ``1/sqrt(i+1)``. Gamma is centred at 1 (stored as an offset
    from zero) so that weight decay pulls it toward 1 rather than 0.
    """

    def __init__(self, channels: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * ((self.gamma + 1.0) * self.scale) + self.beta


class GlobalPoolBias(nn.Module):
    """Board-wide context folded back in as a per-channel bias.

    KataGo pools mean, a board-size-scaled mean, and max; our board is a fixed
    9x9 with no masking, so the scaled mean is a constant multiple of the mean
    and only mean and max carry information.
    """

    def __init__(self, pooled_channels: int, output_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * pooled_channels, output_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=(2, 3))
        maximum = inputs.amax(dim=(2, 3))
        bias = self.linear(torch.cat((mean, maximum), dim=1))
        return bias.unsqueeze(-1).unsqueeze(-1)


class PreActivationBlock(nn.Module):
    """Pre-activation residual block with fixed-scale normalisation."""

    def __init__(self, channels: int, scale: float, gpool_channels: int = 0) -> None:
        super().__init__()
        self.gpool_channels = gpool_channels
        self.channels = channels
        self.norm1 = FixedScaleNorm(channels, scale)
        self.conv1 = nn.Conv2d(
            channels, channels + gpool_channels, kernel_size=3, padding=1, bias=False
        )
        self.gpool = (
            GlobalPoolBias(gpool_channels, channels) if gpool_channels else None
        )
        self.norm2 = FixedScaleNorm(channels, 1.0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.conv1(torch.relu(self.norm1(inputs)))
        if self.gpool is not None:
            regular, pooled = outputs.split([self.channels, self.gpool_channels], dim=1)
            outputs = regular + self.gpool(pooled)
        outputs = self.conv2(torch.relu(self.norm2(outputs)))
        return inputs + outputs


class HeadSet(nn.Module):
    """One independent copy of the policy, value and distance heads."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Conv2d(config.channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, config.action_size),
        )
        self.value_features = nn.Sequential(
            nn.Conv2d(config.channels, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 9 * 9, config.value_hidden),
            nn.ReLU(inplace=True),
        )
        self.value_output = nn.Linear(config.value_hidden, 1)
        self.distance_output = (
            nn.Linear(config.value_hidden, 2) if config.distance_head else None
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        value_features = self.value_features(features)
        value = torch.tanh(self.value_output(value_features)).squeeze(-1)
        distances = (
            self.distance_output(value_features)
            if self.distance_output is not None
            else None
        )
        return self.policy(features), value, distances


class PolicyValueNet(nn.Module):
    """A configurable AlphaZero-style residual policy-value network."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        config = config or NetworkConfig()
        self.config = config
        if config.norm_kind == "fixscaleonenorm":
            self._build_fixed_scale(config)
            self.apply(_initialize)
            self._initialize_fixed_scale()
            return
        if config.norm_kind != "bnorm":
            raise ValueError(f"unsupported norm_kind: {config.norm_kind}")
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
        self.distance_output = (
            nn.Linear(config.value_hidden, 2) if config.distance_head else None
        )
        self.apply(_initialize)

    def _build_fixed_scale(self, config: NetworkConfig) -> None:
        """KataGo's fixed-variance trunk with one BatchNorm and dual heads."""
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.input_planes,
                config.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        )
        self.tower = nn.Sequential(
            *(
                PreActivationBlock(
                    config.channels,
                    scale=1.0 / math.sqrt(index + 1.0),
                    gpool_channels=(
                        config.gpool_channels
                        if (index + 1) in config.gpool_blocks
                        else 0
                    ),
                )
                for index in range(config.blocks)
            )
        )
        trunk_scale = 1.0 / math.sqrt(config.blocks + 1.0)
        # The single BatchNorm: it carries most of the loss, which is what
        # removes the incentive for trunk weights to drift in magnitude.
        self.trunk_bn = nn.BatchNorm2d(config.channels)
        self.trunk_scale = FixedScaleNorm(config.channels, trunk_scale)
        self.bn_heads = HeadSet(config)
        self.inference_heads = HeadSet(config)

    def _initialize_fixed_scale(self) -> None:
        """Init corrections the fixed-scale variance budget depends on.

        The trunk convs keep ReLU gain, which is what makes a pre-activation
        conv-activation pair variance preserving. Two places must not:
        the stem has no activation in front of it, so ReLU gain would hand the
        trunk variance 2 instead of 1; and a pooled bias added into a
        variance-1 signal starts at zero so global pooling learns in rather
        than doubling the variance at initialisation.
        """
        stem_conv = self.stem[0]
        assert isinstance(stem_conv, nn.Conv2d)
        nn.init.kaiming_normal_(stem_conv.weight, nonlinearity="linear")
        for module in self.modules():
            if isinstance(module, GlobalPoolBias):
                nn.init.zeros_(module.linear.weight)
                nn.init.zeros_(module.linear.bias)

    @property
    def _is_fixed_scale(self) -> bool:
        return self.config.norm_kind == "fixscaleonenorm"

    def _trunk(self, inputs: Tensor) -> Tensor:
        return self.tower(self.stem(inputs))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if self._is_fixed_scale:
            features = torch.relu(self.trunk_scale(self._trunk(inputs)))
            policy_logits, value, _ = self.inference_heads(features)
            return policy_logits, value
        features = self.tower(self.stem(inputs))
        policy_logits = self.policy_head(features)
        value = torch.tanh(self.value_output(self.value_features(features)))
        return policy_logits, value.squeeze(-1)

    def forward_train_heads(
        self, inputs: Tensor
    ) -> list[tuple[float, Tensor, Tensor, Tensor | None]]:
        """Every head set that takes loss, paired with its loss weight.

        The inference head is always last, so callers can report its losses as
        the comparable ones regardless of architecture.
        """
        if not self._is_fixed_scale:
            return [(1.0, *self.forward_train(inputs))]
        trunk = self._trunk(inputs)
        bn_weight = self.config.bn_head_loss_weight
        inference = self.inference_heads(torch.relu(self.trunk_scale(trunk)))
        return [
            (bn_weight, *self.bn_heads(torch.relu(self.trunk_bn(trunk)))),
            (1.0 - bn_weight, *inference),
        ]

    def forward_train(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Forward pass that also yields auxiliary distance predictions."""
        if self._is_fixed_scale:
            features = torch.relu(self.trunk_scale(self._trunk(inputs)))
            return self.inference_heads(features)
        features = self.tower(self.stem(inputs))
        policy_logits = self.policy_head(features)
        value_features = self.value_features(features)
        value = torch.tanh(self.value_output(value_features))
        distances = (
            self.distance_output(value_features)
            if self.distance_output is not None
            else None
        )
        return policy_logits, value.squeeze(-1), distances


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


@dataclass(slots=True)
class _HostSlot:
    """Pinned host staging buffers for one in-flight evaluation batch."""

    inputs: Tensor
    logits: Tensor
    values: Tensor


@dataclass(slots=True)
class PendingPlanes:
    """Handle to a submitted, not-yet-collected evaluation batch.

    The numpy views returned by ``collect_planes`` alias the pinned slot and
    stay valid until the slot is reused, two submissions later; consumers must
    copy or serialize them before then (the eval server pickles them into
    worker pipes immediately, which copies).
    """

    slot: _HostSlot
    count: int
    event: Any

    def ready(self) -> bool:
        return bool(self.event.query())


class TorchEvaluator:
    """Batched, inference-only adapter used by MCTS."""

    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        *,
        amp: bool = True,
        jit_freeze: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        # Gate bf16 autocast on Ampere+ (sm80), same as training: Pascal only
        # emulates bf16, and its cublasLt bf16 kernels can hang inside
        # cuLaunchKernel on driver 580 (see docs/node-1080-wedge-log.md).
        self.amp = (
            amp
            and device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] >= 8
        )
        # Async submit/collect lets the eval server overlap GPU compute with
        # queue draining and reply serialization; CUDA only.
        self.supports_async_planes = device.type == "cuda"
        self._slots: list[_HostSlot] = []
        self._slot_index = 0
        self.model.eval()
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self._forward: Any = self.model
        # On pre-Ampere CUDA (fp32 inference), freeze a traced copy:
        # optimize_for_inference folds batchnorm and NNC-fuses the
        # elementwise/SE chains, measured +11% samples/s on the GTX 1080
        # (max logit deviation 1.8e-4, same order as batchnorm folding).
        # Autocast does not apply inside TorchScript, so the bf16 path
        # keeps the eager module.
        if jit_freeze and device.type == "cuda" and not self.amp:
            example = torch.zeros(
                (2, self.model.config.input_planes, 9, 9), device=device
            )
            with torch.no_grad():
                traced = torch.jit.trace(self.model, example)
                self._forward = torch.jit.optimize_for_inference(
                    torch.jit.freeze(traced.eval())
                )

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
            logits, values = self._forward(inputs)
        return (
            logits.float().cpu().numpy(),
            values.float().cpu().numpy(),
        )

    def _staging_slot(self, count: int) -> _HostSlot:
        """Rotate between two pinned slots, growing capacity as needed."""
        if len(self._slots) < 2:
            self._slots.append(self._new_slot(count))
            self._slot_index = len(self._slots) - 1
            return self._slots[self._slot_index]
        self._slot_index = (self._slot_index + 1) % 2
        if self._slots[self._slot_index].inputs.shape[0] < count:
            self._slots[self._slot_index] = self._new_slot(count)
        return self._slots[self._slot_index]

    def _new_slot(self, count: int) -> _HostSlot:
        capacity = max(256, 1 << (count - 1).bit_length())
        planes = self.model.config.input_planes
        return _HostSlot(
            inputs=torch.empty(
                (capacity, planes, 9, 9), dtype=torch.float32, pin_memory=True
            ),
            logits=torch.empty(
                (capacity, self.model.config.action_size),
                dtype=torch.float32,
                pin_memory=True,
            ),
            values=torch.empty((capacity,), dtype=torch.float32, pin_memory=True),
        )

    def submit_planes(self, planes: FloatArray) -> PendingPlanes:
        """Launch one evaluation batch without waiting for the result.

        The H2D copy, forward pass, and D2H copy are all queued on the CUDA
        stream through pinned staging buffers, so the caller keeps the CPU
        while the GPU works; ``collect_planes`` waits and returns the arrays.
        """
        count = len(planes)
        slot = self._staging_slot(count)
        slot.inputs[:count].copy_(torch.from_numpy(planes))
        inputs = slot.inputs[:count].to(self.device, non_blocking=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.amp
            else nullcontext()
        )
        with torch.inference_mode(), context:
            logits, values = self._forward(inputs)
            slot.logits[:count].copy_(logits.float(), non_blocking=True)
            slot.values[:count].copy_(values.float(), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return PendingPlanes(slot=slot, count=count, event=event)

    def collect_planes(self, pending: PendingPlanes) -> tuple[FloatArray, FloatArray]:
        pending.event.synchronize()
        return (
            pending.slot.logits[: pending.count].numpy(),
            pending.slot.values[: pending.count].numpy(),
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

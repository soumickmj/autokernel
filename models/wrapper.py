"""
ModelWrapper -- Universal wrapper for any nn.Module with flexible input specification.

Wraps any PyTorch nn.Module (text models, image models, diffusion models, etc.)
so that AutoKernel can profile, extract, and verify kernels regardless of the
model's input/output conventions.

Usage:
    from models.wrapper import ModelWrapper, InputSpec

    # Wrap a custom model with explicit input specification
    spec = InputSpec.image_2d(batch=4, channels=3, height=256, width=256)
    wrapped = ModelWrapper(my_model, input_spec=spec)

    # Or let AutoKernel infer the input type
    wrapped = ModelWrapper(my_model, input_shape=[4, 3, 256, 256])

    # Use with profiler / verifier
    uv run profile.py --model models/my_model.py --class-name MyModel --input-shape 4,3,256,256
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

class InputType(Enum):
    """Supported input data types for models."""
    TOKEN_IDS = auto()      # Integer token IDs for language models
    IMAGE_2D = auto()       # 2D images: (B, C, H, W)
    IMAGE_3D = auto()       # 3D volumetric images: (B, C, D, H, W)
    FLOAT_TENSOR = auto()   # Generic float tensor of any shape
    MULTIPLE = auto()       # Multiple named tensors (e.g. encoder-decoder)


# ---------------------------------------------------------------------------
# InputSpec -- describes what a model expects as input
# ---------------------------------------------------------------------------

@dataclass
class InputSpec:
    """
    Describes the expected input format for a model.

    Attributes:
        input_type:    The kind of input data (token IDs, 2D images, etc.).
        shape:         Default shape for the input tensor(s), e.g. (B, C, H, W).
        dtype:         Default dtype for the tensor(s).
        names:         Optional dict mapping argument names to shapes, for models
                       that accept multiple named tensors.
        value_range:   (min, max) range for generated input values.  For token
                       IDs this is (0, vocab_size); for images typically (0, 1)
                       or (-1, 1).
        channels_last: Whether to generate tensors in channels-last memory format.
        forward_key:   If the model's forward() expects a keyword argument
                       (e.g. ``input_ids``), specify it here.  When ``None``,
                       the tensor is passed as a positional argument.
    """

    input_type: InputType = InputType.FLOAT_TENSOR
    shape: Tuple[int, ...] = ()
    dtype: torch.dtype = torch.float32
    names: Optional[Dict[str, Tuple[int, ...]]] = None
    value_range: Tuple[float, float] = (-1.0, 1.0)
    channels_last: bool = False
    forward_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def token_ids(
        cls,
        batch: int = 1,
        seq_len: int = 512,
        vocab_size: int = 32000,
    ) -> "InputSpec":
        """Create an InputSpec for language models (integer token IDs)."""
        return cls(
            input_type=InputType.TOKEN_IDS,
            shape=(batch, seq_len),
            dtype=torch.long,
            value_range=(0, vocab_size),
            forward_key="input_ids",
        )

    @classmethod
    def image_2d(
        cls,
        batch: int = 1,
        channels: int = 3,
        height: int = 224,
        width: int = 224,
        dtype: torch.dtype = torch.float32,
        value_range: Tuple[float, float] = (0.0, 1.0),
    ) -> "InputSpec":
        """Create an InputSpec for 2D image models (B, C, H, W)."""
        return cls(
            input_type=InputType.IMAGE_2D,
            shape=(batch, channels, height, width),
            dtype=dtype,
            value_range=value_range,
        )

    @classmethod
    def image_3d(
        cls,
        batch: int = 1,
        channels: int = 1,
        depth: int = 64,
        height: int = 64,
        width: int = 64,
        dtype: torch.dtype = torch.float32,
        value_range: Tuple[float, float] = (0.0, 1.0),
    ) -> "InputSpec":
        """Create an InputSpec for 3D volumetric models (B, C, D, H, W)."""
        return cls(
            input_type=InputType.IMAGE_3D,
            shape=(batch, channels, depth, height, width),
            dtype=dtype,
            value_range=value_range,
        )

    @classmethod
    def float_tensor(
        cls,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> "InputSpec":
        """Create an InputSpec for a generic float tensor."""
        return cls(
            input_type=InputType.FLOAT_TENSOR,
            shape=shape,
            dtype=dtype,
        )

    @classmethod
    def multiple(
        cls,
        names: Dict[str, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32,
    ) -> "InputSpec":
        """
        Create an InputSpec for models expecting multiple named tensors.

        Example::

            spec = InputSpec.multiple({
                "x": (4, 3, 256, 256),
                "timestep": (4,),
            })
        """
        # Use the first tensor's shape as the canonical shape
        first_shape = next(iter(names.values()))
        return cls(
            input_type=InputType.MULTIPLE,
            shape=first_shape,
            dtype=dtype,
            names=names,
        )

    # ------------------------------------------------------------------
    # Input generation
    # ------------------------------------------------------------------

    def generate(
        self,
        device: str = "cuda",
        seed: int = 42,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate a sample input tensor (or dict of tensors) on *device*.

        Returns:
            A single ``torch.Tensor`` for simple input types, or a
            ``dict[str, torch.Tensor]`` for TOKEN_IDS (keyed by
            ``forward_key``) or MULTIPLE inputs.
        """
        torch.manual_seed(seed)
        lo, hi = self.value_range

        if self.input_type == InputType.TOKEN_IDS:
            t = torch.randint(int(lo), int(hi), self.shape, device=device, dtype=torch.long)
            key = self.forward_key or "input_ids"
            return {key: t}

        if self.input_type == InputType.MULTIPLE:
            assert self.names is not None
            result: Dict[str, torch.Tensor] = {}
            for name, shape in self.names.items():
                result[name] = torch.randn(*shape, device=device, dtype=self.dtype) * (hi - lo) + lo
            return result

        # IMAGE_2D, IMAGE_3D, FLOAT_TENSOR
        t = torch.randn(*self.shape, device=device, dtype=self.dtype)
        # Scale to value_range if not standard normal
        if (lo, hi) != (-1.0, 1.0):
            t = t.clamp(-3, 3)  # avoid extreme values
            t = (t + 3) / 6  # map to [0, 1]
            t = t * (hi - lo) + lo
        if self.channels_last and t.dim() == 4:
            t = t.to(memory_format=torch.channels_last)
        return t


# ---------------------------------------------------------------------------
# ModelWrapper -- wraps any nn.Module for use with AutoKernel
# ---------------------------------------------------------------------------

class ModelWrapper(nn.Module):
    """
    Universal model wrapper for AutoKernel profiling and optimization.

    Wraps any ``nn.Module`` and provides:
    - Automatic or explicit input specification via ``InputSpec``
    - Consistent forward-pass calling convention
    - Model introspection helpers (parameter count, layer types, etc.)

    The wrapper is transparent: ``wrapped_model(x)`` calls the inner
    model's ``forward()`` with the correct argument passing convention.

    Examples::

        # Wrap with automatic detection
        wrapped = ModelWrapper(my_cnn, input_shape=[4, 3, 224, 224])

        # Wrap with explicit spec
        spec = InputSpec.image_2d(batch=4)
        wrapped = ModelWrapper(my_cnn, input_spec=spec)

        # Wrap a diffusion model that takes (x, timestep)
        spec = InputSpec.multiple({"x": (4, 3, 64, 64), "timestep": (4,)})
        wrapped = ModelWrapper(diffusion_model, input_spec=spec)
    """

    def __init__(
        self,
        model: nn.Module,
        input_spec: Optional[InputSpec] = None,
        input_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch.dtype] = None,
        forward_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model:       The inner ``nn.Module`` to wrap.
            input_spec:  Explicit ``InputSpec`` describing input format.
                         If *None*, the wrapper will try to infer it from
                         ``input_shape`` and the model's structure.
            input_shape: Fallback shape used when ``input_spec`` is None.
            dtype:       Override dtype.  Only used when ``input_spec`` is None.
            forward_fn:  Optional custom callable that performs the forward pass.
                         Signature: ``forward_fn(model, *args, **kwargs) -> output``.
                         Useful for models with non-standard forward signatures
                         (e.g. diffusion models needing a timestep argument).
        """
        super().__init__()
        self.inner = model
        self.forward_fn = forward_fn

        if input_spec is not None:
            self.input_spec = input_spec
        elif input_shape is not None:
            self.input_spec = _infer_input_spec(model, list(input_shape), dtype)
        else:
            # Absolute fallback: generic float tensor
            self.input_spec = InputSpec.float_tensor(shape=(1,))

    # ------------------------------------------------------------------
    # Forward pass -- delegate to inner model
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.forward_fn is not None:
            return self.forward_fn(self.inner, *args, **kwargs)
        return self.inner(*args, **kwargs)

    # ------------------------------------------------------------------
    # Convenience: generate sample input for this model
    # ------------------------------------------------------------------

    def generate_sample_input(
        self,
        device: str = "cuda",
        seed: int = 42,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate sample input matching the stored ``InputSpec``."""
        return self.input_spec.generate(device=device, seed=seed)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.inner.parameters())

    def layer_summary(self) -> Dict[str, int]:
        """Count modules by type (e.g. {'Linear': 12, 'Conv2d': 5, ...})."""
        counts: Dict[str, int] = {}
        for _, m in self.inner.named_modules():
            name = type(m).__name__
            counts[name] = counts.get(name, 0) + 1
        return counts

    def has_conv2d(self) -> bool:
        return any(isinstance(m, nn.Conv2d) for m in self.inner.modules())

    def has_conv3d(self) -> bool:
        return any(isinstance(m, nn.Conv3d) for m in self.inner.modules())

    def has_batchnorm(self) -> bool:
        return any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in self.inner.modules()
        )

    def has_embedding(self) -> bool:
        return any(isinstance(m, nn.Embedding) for m in self.inner.modules())


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_input_spec(
    model: nn.Module,
    input_shape: List[int],
    dtype: Optional[torch.dtype] = None,
) -> InputSpec:
    """
    Infer an ``InputSpec`` from the model's structure and a user-supplied shape.

    Heuristics:
    - If the model has an ``nn.Embedding`` first child → token IDs
    - If shape is (B, C, H, W) and model has ``nn.Conv2d`` → 2D image
    - If shape is (B, C, D, H, W) and model has ``nn.Conv3d`` → 3D image
    - Otherwise → generic float tensor
    """
    resolved_dtype = dtype or torch.float32

    # Check for language model (embedding layer)
    if _has_embedding_first(model):
        batch = input_shape[0] if len(input_shape) >= 1 else 1
        seq_len = input_shape[1] if len(input_shape) >= 2 else 512
        return InputSpec.token_ids(batch=batch, seq_len=seq_len)

    # Check for 3D volumetric model
    if len(input_shape) == 5 and _has_conv3d(model):
        return InputSpec.image_3d(
            batch=input_shape[0],
            channels=input_shape[1],
            depth=input_shape[2],
            height=input_shape[3],
            width=input_shape[4],
            dtype=resolved_dtype,
        )

    # Check for 2D image model
    if len(input_shape) == 4 and _has_conv2d(model):
        return InputSpec.image_2d(
            batch=input_shape[0],
            channels=input_shape[1],
            height=input_shape[2],
            width=input_shape[3],
            dtype=resolved_dtype,
        )

    # Generic float tensor
    return InputSpec.float_tensor(shape=tuple(input_shape), dtype=resolved_dtype)


def infer_input_type(model: nn.Module) -> InputType:
    """
    Infer the most likely ``InputType`` from a model's structure.

    This is a lightweight heuristic used when no ``InputSpec`` is provided.
    """
    # Check for embedding → language model
    if _has_embedding_first(model):
        return InputType.TOKEN_IDS

    # Check class name heuristics
    cls_name = type(model).__name__.lower()
    lm_keywords = [
        "causal", "lm", "gpt", "llama", "bert", "t5", "opt", "falcon",
        "mistral", "gemma", "phi", "qwen", "codegen", "bloom", "mpt",
    ]
    if any(kw in cls_name for kw in lm_keywords):
        return InputType.TOKEN_IDS

    # Check for Conv3d → volumetric
    if _has_conv3d(model):
        return InputType.IMAGE_3D

    # Check for Conv2d → 2D image
    if _has_conv2d(model):
        return InputType.IMAGE_2D

    # Fallback
    return InputType.FLOAT_TENSOR


def _has_embedding_first(model: nn.Module) -> bool:
    """Check if the model's first child is an embedding layer."""
    for name, child in model.named_children():
        if isinstance(child, nn.Embedding):
            return True
        child_name = type(child).__name__.lower()
        if "embed" in name.lower() or "embedding" in child_name:
            return True
        break  # only check first child
    # Also check forward signature
    try:
        sig = inspect.signature(model.forward)
        if "input_ids" in sig.parameters:
            return True
    except (ValueError, TypeError):
        pass
    return False


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model contains any Conv2d layers."""
    return any(isinstance(m, nn.Conv2d) for m in model.modules())


def _has_conv3d(model: nn.Module) -> bool:
    """Check if the model contains any Conv3d layers."""
    return any(isinstance(m, nn.Conv3d) for m in model.modules())

"""
Tests for the new image model support and ModelWrapper functionality.

These tests validate:
1. InputSpec creation for different input types (2D images, 3D images, tokens, etc.)
2. ModelWrapper wrapping any nn.Module
3. Input type inference for image vs. text models
4. Conv2d/Conv3d/BatchNorm2d kernel replacement wrappers
5. Image model examples (DiffusionAutoencoder, UNet2D, SimpleResNet, VolumetricSegNet)
6. Reference implementations for conv2d, conv3d, batchnorm2d
7. Kernel classification for image operations
8. Extract shape keys for conv2d, conv3d, batchnorm2d
"""

import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Test InputSpec and InputType
# ---------------------------------------------------------------------------

def test_input_spec_token_ids():
    from models.wrapper import InputSpec, InputType

    spec = InputSpec.token_ids(batch=2, seq_len=128, vocab_size=50000)
    assert spec.input_type == InputType.TOKEN_IDS
    assert spec.shape == (2, 128)
    assert spec.dtype == torch.long
    assert spec.value_range == (0, 50000)
    assert spec.forward_key == "input_ids"
    print("PASS: test_input_spec_token_ids")


def test_input_spec_image_2d():
    from models.wrapper import InputSpec, InputType

    spec = InputSpec.image_2d(batch=4, channels=3, height=256, width=256)
    assert spec.input_type == InputType.IMAGE_2D
    assert spec.shape == (4, 3, 256, 256)
    assert spec.dtype == torch.float32
    assert spec.value_range == (0.0, 1.0)
    print("PASS: test_input_spec_image_2d")


def test_input_spec_image_3d():
    from models.wrapper import InputSpec, InputType

    spec = InputSpec.image_3d(batch=1, channels=1, depth=32, height=32, width=32)
    assert spec.input_type == InputType.IMAGE_3D
    assert spec.shape == (1, 1, 32, 32, 32)
    assert spec.dtype == torch.float32
    print("PASS: test_input_spec_image_3d")


def test_input_spec_float_tensor():
    from models.wrapper import InputSpec, InputType

    spec = InputSpec.float_tensor(shape=(8, 128))
    assert spec.input_type == InputType.FLOAT_TENSOR
    assert spec.shape == (8, 128)
    print("PASS: test_input_spec_float_tensor")


def test_input_spec_multiple():
    from models.wrapper import InputSpec, InputType

    spec = InputSpec.multiple({
        "x": (4, 3, 64, 64),
        "timestep": (4,),
    })
    assert spec.input_type == InputType.MULTIPLE
    assert spec.names is not None
    assert "x" in spec.names
    assert "timestep" in spec.names
    print("PASS: test_input_spec_multiple")


def test_input_spec_generate_token_ids():
    from models.wrapper import InputSpec

    spec = InputSpec.token_ids(batch=2, seq_len=64, vocab_size=32000)
    result = spec.generate(device="cpu")
    assert isinstance(result, dict)
    assert "input_ids" in result
    assert result["input_ids"].shape == (2, 64)
    assert result["input_ids"].dtype == torch.long
    assert (result["input_ids"] >= 0).all()
    assert (result["input_ids"] < 32000).all()
    print("PASS: test_input_spec_generate_token_ids")


def test_input_spec_generate_image_2d():
    from models.wrapper import InputSpec

    spec = InputSpec.image_2d(batch=2, channels=3, height=64, width=64)
    result = spec.generate(device="cpu")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3, 64, 64)
    assert result.dtype == torch.float32
    print("PASS: test_input_spec_generate_image_2d")


def test_input_spec_generate_image_3d():
    from models.wrapper import InputSpec

    spec = InputSpec.image_3d(batch=1, channels=1, depth=16, height=16, width=16)
    result = spec.generate(device="cpu")
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 1, 16, 16, 16)
    print("PASS: test_input_spec_generate_image_3d")


def test_input_spec_generate_multiple():
    from models.wrapper import InputSpec

    spec = InputSpec.multiple({
        "x": (2, 3, 32, 32),
        "t": (2,),
    })
    result = spec.generate(device="cpu")
    assert isinstance(result, dict)
    assert "x" in result
    assert result["x"].shape == (2, 3, 32, 32)
    assert "t" in result
    assert result["t"].shape == (2,)
    print("PASS: test_input_spec_generate_multiple")


# ---------------------------------------------------------------------------
# 2. Test ModelWrapper
# ---------------------------------------------------------------------------

class _SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class _SimpleLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)
        self.fc = nn.Linear(64, 1000)

    def forward(self, input_ids):
        return self.fc(self.embed(input_ids))


class _Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_model_wrapper_with_cnn():
    from models.wrapper import ModelWrapper, InputSpec

    model = _SimpleCNN()
    spec = InputSpec.image_2d(batch=2, channels=3, height=32, width=32)
    wrapped = ModelWrapper(model, input_spec=spec)

    # Test forward pass
    inp = spec.generate(device="cpu")
    out = wrapped(inp)
    assert out.shape == (2, 10)
    print("PASS: test_model_wrapper_with_cnn")


def test_model_wrapper_infer_from_shape_cnn():
    from models.wrapper import ModelWrapper, InputType

    model = _SimpleCNN()
    wrapped = ModelWrapper(model, input_shape=[2, 3, 32, 32])
    assert wrapped.input_spec.input_type == InputType.IMAGE_2D
    assert wrapped.input_spec.shape == (2, 3, 32, 32)
    print("PASS: test_model_wrapper_infer_from_shape_cnn")


def test_model_wrapper_infer_from_shape_lm():
    from models.wrapper import ModelWrapper, InputType

    model = _SimpleLM()
    wrapped = ModelWrapper(model, input_shape=[2, 64])
    assert wrapped.input_spec.input_type == InputType.TOKEN_IDS
    print("PASS: test_model_wrapper_infer_from_shape_lm")


def test_model_wrapper_infer_from_shape_3d():
    from models.wrapper import ModelWrapper, InputType

    model = _Simple3DCNN()
    wrapped = ModelWrapper(model, input_shape=[1, 1, 16, 16, 16])
    assert wrapped.input_spec.input_type == InputType.IMAGE_3D
    print("PASS: test_model_wrapper_infer_from_shape_3d")


def test_model_wrapper_layer_summary():
    from models.wrapper import ModelWrapper, InputSpec

    model = _SimpleCNN()
    spec = InputSpec.image_2d(batch=2)
    wrapped = ModelWrapper(model, input_spec=spec)

    summary = wrapped.layer_summary()
    assert summary.get("Conv2d", 0) >= 1
    assert summary.get("Linear", 0) >= 1
    assert wrapped.has_conv2d()
    assert not wrapped.has_conv3d()
    assert not wrapped.has_embedding()
    print("PASS: test_model_wrapper_layer_summary")


def test_model_wrapper_param_count():
    from models.wrapper import ModelWrapper, InputSpec

    model = _SimpleMLP()
    spec = InputSpec.float_tensor(shape=(4, 128))
    wrapped = ModelWrapper(model, input_spec=spec)
    assert wrapped.param_count() > 0
    print("PASS: test_model_wrapper_param_count")


def test_model_wrapper_custom_forward_fn():
    from models.wrapper import ModelWrapper, InputSpec

    class _DiffusionLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x, timestep=None):
            if timestep is not None:
                x = x + timestep.view(-1, 1, 1, 1)
            return self.conv(x)

    model = _DiffusionLike()

    def custom_fwd(m, x):
        t = torch.ones(x.shape[0], device=x.device)
        return m(x, timestep=t)

    spec = InputSpec.image_2d(batch=2, channels=3, height=16, width=16)
    wrapped = ModelWrapper(model, input_spec=spec, forward_fn=custom_fwd)

    inp = spec.generate(device="cpu")
    out = wrapped(inp)
    assert out.shape == (2, 3, 16, 16)
    print("PASS: test_model_wrapper_custom_forward_fn")


# ---------------------------------------------------------------------------
# 3. Test infer_input_type in both profile.py and verify.py
# ---------------------------------------------------------------------------

def test_profile_input_inference():
    from profile import _is_language_model, _is_image_model, _is_3d_model

    assert _is_language_model(_SimpleLM()) is True
    assert _is_language_model(_SimpleCNN()) is False

    assert _is_image_model(_SimpleCNN()) is True
    assert _is_image_model(_SimpleMLP()) is False

    assert _is_3d_model(_Simple3DCNN()) is True
    assert _is_3d_model(_SimpleCNN()) is False
    print("PASS: test_profile_input_inference")


def test_verify_infer_input_type():
    from verify import infer_input_type

    assert infer_input_type(_SimpleLM()) == "token_ids"
    assert infer_input_type(_SimpleCNN()) == "image_2d"
    assert infer_input_type(_Simple3DCNN()) == "image_3d"
    assert infer_input_type(_SimpleMLP()) == "float"
    print("PASS: test_verify_infer_input_type")


# ---------------------------------------------------------------------------
# 4. Test profile.py input generation
# ---------------------------------------------------------------------------

def test_profile_generate_input_lm():
    from profile import generate_input

    model = _SimpleLM()
    result = generate_input(model, [2, 64], torch.float32, "cpu")
    assert "input_ids" in result
    assert result["input_ids"].shape == (2, 64)
    assert result["input_ids"].dtype == torch.long
    print("PASS: test_profile_generate_input_lm")


def test_profile_generate_input_cnn():
    from profile import generate_input

    model = _SimpleCNN()
    result = generate_input(model, [2, 3, 32, 32], torch.float32, "cpu")
    assert "x" in result
    assert result["x"].shape == (2, 3, 32, 32)
    assert result["x"].dtype == torch.float32
    print("PASS: test_profile_generate_input_cnn")


def test_profile_generate_input_3d():
    from profile import generate_input

    model = _Simple3DCNN()
    result = generate_input(model, [1, 1, 16, 16, 16], torch.float32, "cpu")
    assert "x" in result
    assert result["x"].shape == (1, 1, 16, 16, 16)
    print("PASS: test_profile_generate_input_3d")


# ---------------------------------------------------------------------------
# 5. Test kernel classification for image ops
# ---------------------------------------------------------------------------

def test_kernel_classification_conv2d():
    from profile import classify_kernel

    assert classify_kernel("cudnn_conv2d_forward") == "conv2d"
    assert classify_kernel("conv_2d_nchw") == "conv2d"
    assert classify_kernel("conv_transpose2d") == "conv2d"
    print("PASS: test_kernel_classification_conv2d")


def test_kernel_classification_conv3d():
    from profile import classify_kernel

    assert classify_kernel("cudnn3d_conv_forward") == "conv3d"
    assert classify_kernel("conv3d_ndhwc") == "conv3d"
    assert classify_kernel("conv_3d_depthwise") == "conv3d"
    print("PASS: test_kernel_classification_conv3d")


def test_kernel_classification_batchnorm():
    from profile import classify_kernel

    assert classify_kernel("cudnn_batch_norm_forward") == "batchnorm2d"
    assert classify_kernel("batchnorm_fwd") == "batchnorm2d"
    print("PASS: test_kernel_classification_batchnorm")


def test_kernel_classification_existing():
    """Ensure existing classifications still work."""
    from profile import classify_kernel

    assert classify_kernel("fmha_v2_flash_attention_fp16_kernel") == "flash_attention"
    assert classify_kernel("cublas_sgemm") == "matmul"
    assert classify_kernel("softmax_forward") == "softmax"
    assert classify_kernel("layernorm_kernel") == "layernorm"
    assert classify_kernel("rmsnorm_kernel") == "rmsnorm"
    print("PASS: test_kernel_classification_existing")


# ---------------------------------------------------------------------------
# 6. Test reference implementations
# ---------------------------------------------------------------------------

def test_reference_conv2d():
    from reference import conv2d_ref

    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(16, 3, 3, 3)
    out = conv2d_ref(x, w, padding=1)
    assert out.shape == (2, 16, 8, 8)

    # With bias
    b = torch.randn(16)
    out_b = conv2d_ref(x, w, bias=b, padding=1)
    assert out_b.shape == (2, 16, 8, 8)
    print("PASS: test_reference_conv2d")


def test_reference_conv3d():
    from reference import conv3d_ref

    x = torch.randn(1, 1, 8, 8, 8)
    w = torch.randn(4, 1, 3, 3, 3)
    out = conv3d_ref(x, w, padding=1)
    assert out.shape == (1, 4, 8, 8, 8)
    print("PASS: test_reference_conv3d")


def test_reference_batchnorm2d():
    from reference import batchnorm2d_ref

    x = torch.randn(2, 8, 16, 16)
    mean = torch.zeros(8)
    var = torch.ones(8)
    weight = torch.ones(8)
    bias = torch.zeros(8)

    out = batchnorm2d_ref(x, mean, var, weight, bias)
    assert out.shape == (2, 8, 16, 16)
    print("PASS: test_reference_batchnorm2d")


# ---------------------------------------------------------------------------
# 7. Test Conv2d/Conv3d/BatchNorm2d replacement wrappers
# ---------------------------------------------------------------------------

def test_conv2d_wrapper():
    from verify import _Conv2dWrapper

    original = nn.Conv2d(3, 16, 3, padding=1)
    kernel_fn = lambda x, w, b=None, s=1, p=0: F.conv2d(x, w, b, stride=s, padding=p)
    wrapper = _Conv2dWrapper(original, kernel_fn)

    x = torch.randn(2, 3, 32, 32)
    out = wrapper(x)
    assert out.shape == (2, 16, 32, 32)
    print("PASS: test_conv2d_wrapper")


def test_conv3d_wrapper():
    from verify import _Conv3dWrapper

    original = nn.Conv3d(1, 4, 3, padding=1)
    kernel_fn = lambda x, w, b=None, s=1, p=0: F.conv3d(x, w, b, stride=s, padding=p)
    wrapper = _Conv3dWrapper(original, kernel_fn)

    x = torch.randn(1, 1, 8, 8, 8)
    out = wrapper(x)
    assert out.shape == (1, 4, 8, 8, 8)
    print("PASS: test_conv3d_wrapper")


def test_batchnorm2d_wrapper():
    from verify import _BatchNorm2dWrapper

    original = nn.BatchNorm2d(16)
    original.eval()
    kernel_fn = lambda x, rm, rv, w=None, b=None, t=False, m=0.1, e=1e-5: \
        F.batch_norm(x, rm, rv, w, b, t, m, e)
    wrapper = _BatchNorm2dWrapper(original, kernel_fn)

    x = torch.randn(2, 16, 8, 8)
    out = wrapper(x)
    assert out.shape == (2, 16, 8, 8)
    print("PASS: test_batchnorm2d_wrapper")


# ---------------------------------------------------------------------------
# 8. Test OptimizedModelContext with conv2d
# ---------------------------------------------------------------------------

def test_optimized_context_conv2d():
    from verify import OptimizedModelContext, KernelReplacement
    import tempfile

    # Create a simple model with Conv2d
    model = _SimpleCNN()

    # Create a trivial optimized kernel file
    kernel_code = '''
import torch
import torch.nn.functional as F

def kernel_fn(x, weight, bias=None, stride=1, padding=0):
    return F.conv2d(x, weight, bias, stride=stride, padding=padding)
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(kernel_code)
        kernel_path = f.name

    try:
        repl = KernelReplacement(
            kernel_type="conv2d",
            rank=1,
            speedup=1.0,
            optimized_path=kernel_path,
        )

        with OptimizedModelContext(model, [repl]) as patched:
            x = torch.randn(2, 3, 32, 32)
            out = patched(x)
            assert out.shape == (2, 10)
    finally:
        os.unlink(kernel_path)

    print("PASS: test_optimized_context_conv2d")


# ---------------------------------------------------------------------------
# 9. Test image model examples instantiation and forward pass
# ---------------------------------------------------------------------------

def test_diffusion_autoencoder():
    from models.image_models import DiffusionAutoencoder

    model = DiffusionAutoencoder(in_channels=3, base_channels=16, latent_channels=4, num_blocks=1)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 32, 32), f"Expected (1, 3, 32, 32), got {out.shape}"
    print("PASS: test_diffusion_autoencoder")


def test_unet2d():
    from models.image_models import UNet2D

    model = UNet2D(in_channels=1, out_channels=1, base_ch=8)
    model.eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 32, 32), f"Expected (1, 1, 32, 32), got {out.shape}"
    print("PASS: test_unet2d")


def test_simple_resnet():
    from models.image_models import SimpleResNet

    model = SimpleResNet(in_channels=3, num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    print("PASS: test_simple_resnet")


def test_volumetric_segnet():
    from models.image_models import VolumetricSegNet

    model = VolumetricSegNet(in_channels=1, out_channels=1, base_ch=4)
    model.eval()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 16, 16, 16), f"Expected (1, 1, 16, 16, 16), got {out.shape}"
    print("PASS: test_volumetric_segnet")


# ---------------------------------------------------------------------------
# 10. Test extract.py shape keys for new types
# ---------------------------------------------------------------------------

def test_extract_shape_keys():
    from extract import SHAPE_KEYS, SHAPE_ALIAS_MAP, TOLERANCES_MAP

    # conv2d
    assert "conv2d" in SHAPE_KEYS
    assert "B" in SHAPE_KEYS["conv2d"]
    assert "conv2d" in SHAPE_ALIAS_MAP
    assert "conv2d" in TOLERANCES_MAP

    # conv3d
    assert "conv3d" in SHAPE_KEYS
    assert "D" in SHAPE_KEYS["conv3d"]
    assert "conv3d" in SHAPE_ALIAS_MAP
    assert "conv3d" in TOLERANCES_MAP

    # batchnorm2d
    assert "batchnorm2d" in SHAPE_KEYS
    assert "batchnorm2d" in SHAPE_ALIAS_MAP
    assert "batchnorm2d" in TOLERANCES_MAP

    print("PASS: test_extract_shape_keys")


def test_extract_flops_and_bytes():
    from extract import FLOPS_FN_SRC, BYTES_FN_SRC, SPEEDUP_ESTIMATES

    for op_type in ["conv2d", "conv3d", "batchnorm2d"]:
        assert op_type in FLOPS_FN_SRC, f"{op_type} missing from FLOPS_FN_SRC"
        assert op_type in BYTES_FN_SRC, f"{op_type} missing from BYTES_FN_SRC"
        assert op_type in SPEEDUP_ESTIMATES, f"{op_type} missing from SPEEDUP_ESTIMATES"

    print("PASS: test_extract_flops_and_bytes")


# ---------------------------------------------------------------------------
# 11. Test wrapper module infer_input_type standalone function
# ---------------------------------------------------------------------------

def test_wrapper_infer_input_type():
    from models.wrapper import infer_input_type, InputType

    assert infer_input_type(_SimpleLM()) == InputType.TOKEN_IDS
    assert infer_input_type(_SimpleCNN()) == InputType.IMAGE_2D
    assert infer_input_type(_Simple3DCNN()) == InputType.IMAGE_3D
    assert infer_input_type(_SimpleMLP()) == InputType.FLOAT_TENSOR
    print("PASS: test_wrapper_infer_input_type")


# ---------------------------------------------------------------------------
# 12. Test roofline estimation for new op types
# ---------------------------------------------------------------------------

def test_roofline_conv2d():
    from profile import estimate_roofline_position
    from profile import GPUSpec

    gpu = GPUSpec()
    assert "compute" in estimate_roofline_position("conv2d_fwd", "conv2d", 100.0, gpu)
    assert "compute" in estimate_roofline_position("conv3d_fwd", "conv3d", 100.0, gpu)
    assert "memory" in estimate_roofline_position("batchnorm_fwd", "batchnorm2d", 10.0, gpu)
    print("PASS: test_roofline_conv2d")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_input_spec_token_ids,
        test_input_spec_image_2d,
        test_input_spec_image_3d,
        test_input_spec_float_tensor,
        test_input_spec_multiple,
        test_input_spec_generate_token_ids,
        test_input_spec_generate_image_2d,
        test_input_spec_generate_image_3d,
        test_input_spec_generate_multiple,
        test_model_wrapper_with_cnn,
        test_model_wrapper_infer_from_shape_cnn,
        test_model_wrapper_infer_from_shape_lm,
        test_model_wrapper_infer_from_shape_3d,
        test_model_wrapper_layer_summary,
        test_model_wrapper_param_count,
        test_model_wrapper_custom_forward_fn,
        test_profile_input_inference,
        test_verify_infer_input_type,
        test_profile_generate_input_lm,
        test_profile_generate_input_cnn,
        test_profile_generate_input_3d,
        test_kernel_classification_conv2d,
        test_kernel_classification_conv3d,
        test_kernel_classification_batchnorm,
        test_kernel_classification_existing,
        test_reference_conv2d,
        test_reference_conv3d,
        test_reference_batchnorm2d,
        test_conv2d_wrapper,
        test_conv3d_wrapper,
        test_batchnorm2d_wrapper,
        test_optimized_context_conv2d,
        test_diffusion_autoencoder,
        test_unet2d,
        test_simple_resnet,
        test_volumetric_segnet,
        test_extract_shape_keys,
        test_extract_flops_and_bytes,
        test_wrapper_infer_input_type,
        test_roofline_conv2d,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")

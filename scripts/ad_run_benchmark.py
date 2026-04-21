#!/usr/bin/env python3
"""
Unified wrapper for running Triton benchmarks with runtime patching.

Zero changes to upstream benchmark files. All customization is done via
environment variables set by the TX YAML config.

Usage:
    python scripts/ad_run_benchmark.py <benchmark_key>

Environment Variables (set by TX via YAML env: blocks):
    AD_TRITON_SIZES         Override x_vals. Python list literal for one shape.
                            Multi-dim: "[1, 1024, 1024, 4096]"
                            Single-dim (softmax): "1024"
    AD_OVERRIDE_CONFIGS     Override FlexAttention fwd autotuner configs.
                            "[FlexConfig(128, 64, 2, 8)]" or "" for defaults.
    AD_FLEX_DECODE_CONFIGS  Override FlexAttention decode autotuner configs.
                            "[FlexDecodeConfig(32, 1, 2)]"
    AD_FLEX_MASKS           Filter mask types for custom_masks benchmark.
                            Comma-separated: "NATTEN,Alibi"
    AD_TRITON_AUTOTUNE_CONFIG
                            Force a specific Triton @triton.autotune config,
                            bypassing the autotuner search. Comma-separated
                            KEY=VALUE pairs. num_stages and num_warps are
                            top-level; all others are kernel kwargs.
                            Example: "BLOCK_SIZE_M=256,BLOCK_SIZE_N=128,
                            BLOCK_SIZE_K=32,GROUP_SIZE_M=4,grf_mode=256,
                            num_stages=2,num_warps=32"
    BATCH_SIZE              FlexAttention batch size (default 1).
    FA_KERNEL_MODE          Forward/backward mode: "fwd" or "bwd".
    INT8_ONLY               gemm_postop_addmatrix int8 mode: "1".
    ALL_DTYPES              gemm_postop_addmatrix both dtypes: "1".
    TRITON_DISABLE_TUNER    Disable autotuning for flash_attention: "1".
    TRITON_INTEL_ADVANCED_PATH  Flash attention advanced path: "1".
    THROUGHPUT_TEST         FlexAttention multi-batch sweep: "1".
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import types

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------
# Each entry maps a benchmark key to its module and access pattern.
#
# "factory" entries: module has get_benchmark(**kwargs) that returns a Mark.
# "attr" entries: module exposes a module-level Mark variable.
# "env_before_import": env vars to set before importing the module.
# "flex_attention": True if FlexAttention config patching applies.

BENCHMARKS = {
    # --- Factory modules (have get_benchmark) ---
    "softmax": {
        "module": "fused_softmax",
        "factory": "get_benchmark",
    },
    "gemm": {
        "module": "gemm_benchmark",
        "factory": "get_benchmark",
    },
    "gemm_bt": {
        "module": "gemm_benchmark",
        "factory": "get_benchmark",
        "kwargs": {"transpose_b": True},
    },
    "gemm_at": {
        "module": "gemm_benchmark",
        "factory": "get_benchmark",
        "kwargs": {"transpose_a": True},
    },
    "gemm_block_ptr": {
        "module": "gemm_block_ptr_benchmark",
        "factory": "get_benchmark",
    },
    "gemm_tensor_of_ptr": {
        "module": "gemm_tensor_of_ptr_benchmark",
        "factory": "get_benchmark",
    },
    "flash_attention_fwd": {
        "module": "flash_attention_benchmark",
        "factory": "get_benchmark",
        "kwargs": {"fa_kernel_mode": "fwd"},
    },
    "flash_attention_bwd": {
        "module": "flash_attention_benchmark",
        "factory": "get_benchmark",
        "kwargs": {"fa_kernel_mode": "bwd"},
    },
    "prefix_sums": {
        "module": "prefix_sums",
        "factory": "get_benchmark",
    },

    # --- Module-level benchmark (no factory) ---
    "gemm_streamk": {
        "module": "gemm_streamk_benchmark",
        "attr": "benchmark",
    },
    "gemm_splitk": {
        "module": "gemm_splitk_benchmark",
        "attr": "benchmark",
    },
    "gemm_preop_exp": {
        "module": "gemm_preop_exp_benchmark",
        "attr": "benchmark",
    },
    "gemm_postop_gelu": {
        "module": "gemm_postop_gelu_benchmark",
        "attr": "benchmark",
    },
    "gemm_postop_addmatrix": {
        "module": "gemm_postop_addmatrix_benchmark",
        "attr": "benchmark",
    },
    "gemm_postop_addmatrix_int8": {
        "module": "gemm_postop_addmatrix_benchmark",
        "attr": "benchmark",
        "env_before_import": {"INT8_ONLY": "1"},
    },

    # --- FlexAttention (module-level benchmark + inductor config patching) ---
    "flex_attn_causal_fwd": {
        "module": "flex_attention_benchmark_causal_mask",
        "attr": "benchmark",
        "flex_attention": True,
    },
    "flex_attn_causal_bwd": {
        "module": "flex_attention_benchmark_causal_mask",
        "attr": "benchmark",
        "flex_attention": True,
        "env_before_import": {"FA_KERNEL_MODE": "bwd"},
    },
    "flex_attn_causal_batch4": {
        "module": "flex_attention_benchmark_causal_mask",
        "attr": "benchmark",
        "flex_attention": True,
        "env_before_import": {"BATCH_SIZE": "4"},
    },
    "flex_attn_custom_masks": {
        "module": "flex_attention_benchmark_custom_masks",
        "attr": "benchmark",
        "flex_attention": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_native_modules():
    """Pre-register mock modules for native extensions that may not be available.

    We only run the triton provider, so xetla_kernel and onednn_kernel are
    never called. But some benchmark modules import them at the top level,
    which would fail if the native extensions aren't built.
    """
    for mod_name in ("xetla_kernel", "onednn_kernel"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)


# Per-arch has_* capability defaults for the cross-compile mock.
#
# Without these, compiler.py's tgt_prop.get('has_2d_block_io', False) etc. all
# default False, so MaterializeBlockPointer / Pipeline / AccelerateMatmul are
# skipped and Triton emits scalar SPIR-V loads + scalar FMAs. See
# cross_compile_capabilities.md for how to verify these match what a real
# driver reports on each arch.
_ARCH_CAPABILITIES = {
    # Xe2 family (Battlemage)
    "bmg": {
        "has_2d_block_io": True,
        "has_subgroup_matrix_multiply_accumulate": True,
        "has_bfloat16_conversion": True,
        "has_bfloat16_arithmetic": True,
        "has_predicated_io": True,
        "has_16bit_atomics": True,
        "has_fp64": True,
    },
    # Xe1 HPC (Ponte Vecchio). Note: TF32 MMA is False on PVC 1100 (test_driver.py).
    "pvc": {
        "has_2d_block_io": True,
        "has_subgroup_matrix_multiply_accumulate": True,
        "has_bfloat16_conversion": True,
        "has_bfloat16_arithmetic": True,
        "has_16bit_atomics": True,
        "has_fp64": True,
    },
    # Xe3P family (CRI). Adds BF8 DPAS and (on some configs) scaled MMA / F4.
    "cri": {
        "has_2d_block_io": True,
        "has_subgroup_matrix_multiply_accumulate": True,
        "has_subgroup_matrix_multiply_accumulate_bfloat8": True,
        "has_bfloat16_conversion": True,
        "has_bfloat16_arithmetic": True,
        "has_predicated_io": True,
        "has_16bit_atomics": True,
        "has_fp64": True,
        "has_256b_prefetch": True,
    },
    # Xe3 (Panther Lake) share Xe3P fast-path capabilities.
    "ptl_h": {
        "has_2d_block_io": True,
        "has_subgroup_matrix_multiply_accumulate": True,
        "has_bfloat16_conversion": True,
        "has_bfloat16_arithmetic": True,
        "has_predicated_io": True,
        "has_16bit_atomics": True,
    },
    "ptl_u": {
        "has_2d_block_io": True,
        "has_subgroup_matrix_multiply_accumulate": True,
        "has_bfloat16_conversion": True,
        "has_bfloat16_arithmetic": True,
        "has_predicated_io": True,
        "has_16bit_atomics": True,
    },
}


def _mock_gpu_for_cross_compile():
    """Mock GPU subsystem for cross-compilation on machines without an Intel GPU.

    Three layers of mocking:
    1. torch.xpu module functions (called at benchmark module import time)
    2. Triton XPUDriver methods (called during JIT compilation)
    3. Tensor factory redirect: device='xpu' → device='cpu'

    Must be called BEFORE importing any benchmark module.
    Controlled by AD_MOCK_GPU=1 env var.
    """
    import functools
    import torch
    from unittest.mock import MagicMock

    arch = os.environ.get("TRITON_INTEL_DEVICE_ARCH", "unknown")
    print(f"[mock_gpu] Mocking GPU subsystem for cross-compilation (arch={arch})")

    # Force XPU backend to avoid "2 active drivers" error when CUDA is
    # also available on the host.
    os.environ.setdefault("TRITON_DEFAULT_BACKEND", "intel")

    # ---- Layer 1: torch.xpu function mocks ----

    # Mock device properties object (returned by torch.xpu.get_device_properties)
    _mock_dev_props = MagicMock()
    _mock_dev_props.total_memory = 128 * 1024**3  # 128 GB
    _mock_dev_props.max_work_group_size = 1024
    _mock_dev_props.name = f"Cross-compile ({arch})"
    _mock_dev_props.gpu_subslice_count = 64  # used by flex_decoding.get_split_k()

    # Mock device capability dict (returned by torch.xpu.get_device_capability)
    _mock_dev_cap = {
        "architecture": 0,  # dummy, overridden by TRITON_INTEL_DEVICE_ARCH
        "device_id": 0,
        "gpu_subslice_count": 64,
        "max_work_group_size": 1024,
        "max_num_sub_groups": 64,
        "sub_group_sizes": [16, 32],
    }

    # Mock stream / event objects
    _mock_stream = MagicMock()
    _mock_stream.sycl_queue = 0  # dummy sycl_queue — never used (launcher is no-op)

    _mock_event = MagicMock()
    _mock_event.elapsed_time = MagicMock(return_value=0.0)

    torch.xpu.is_available = lambda: True
    torch.xpu.device_count = lambda: 1
    torch.xpu.current_device = lambda: 0
    torch.xpu.get_device_name = lambda *a, **kw: f"Cross-compile ({arch})"
    torch.xpu.get_device_properties = lambda *a, **kw: _mock_dev_props
    torch.xpu.get_device_capability = lambda *a, **kw: _mock_dev_cap
    torch.xpu.synchronize = lambda *a, **kw: None
    torch.xpu.empty_cache = lambda *a, **kw: None
    torch.xpu.current_stream = lambda *a, **kw: _mock_stream
    torch.xpu.Event = lambda *a, **kw: _mock_event

    # Prevent _lazy_init from calling torch._C._xpu_init() — it raises
    # "No XPU devices are available" in contexts like preserve_rng_state.
    torch.xpu._lazy_init = lambda: None
    torch.xpu.get_rng_state = lambda *a, **kw: torch.empty(0, dtype=torch.uint8)
    torch.xpu.set_rng_state = lambda *a, **kw: None
    torch.xpu.manual_seed = lambda *a, **kw: None
    torch.xpu.manual_seed_all = lambda *a, **kw: None

    print("[mock_gpu] Layer 1: torch.xpu functions mocked")

    # ---- Layer 2: Triton XPUDriver mocks ----

    from triton.backends.intel.driver import XPUDriver
    from triton.backends.compiler import GPUTarget

    # Build mock device properties dict for get_current_target
    extensions_str = os.environ.get("TRITON_INTEL_DEVICE_EXTENSIONS", "")
    target_props = dict(_mock_dev_cap)
    target_props["arch"] = arch
    target_props["__intel_already_queried_extensions__"] = True
    # Seed has_* capability flags from the per-arch table; env var adds more.
    # Without these, compiler.py's tgt_prop.get('has_*', False) returns False
    # for every capability and Triton falls back to scalar loads + scalar FMAs.
    arch_caps = _ARCH_CAPABILITIES.get(arch, {})
    for cap, enabled in arch_caps.items():
        target_props[cap] = enabled
    if extensions_str:
        for ext in extensions_str.split():
            target_props[ext] = True
    _mock_target = GPUTarget("xpu", target_props, warp_size=32)

    # Mock utils object (prevents __getattr__ from creating XPUUtils which
    # would call init_devices → segfault without GPU)
    class _MockUtils:
        def get_device_properties(self, dev=0):
            return {"max_work_group_size": 1024}
        def get_current_device(self):
            return 0

    # Patch XPUDriver methods
    XPUDriver.get_current_device = lambda self=None: 0
    XPUDriver.get_current_stream = lambda self=None, dev=None: _mock_stream.sycl_queue
    XPUDriver.get_current_target = lambda self=None: _mock_target
    XPUDriver.get_active_torch_device = lambda self=None: torch.device("cpu")

    # Pre-set utils on the class to prevent lazy XPUUtils() init via __getattr__
    # XPUDriver.__getattr__ checks name == "utils" and creates XPUUtils() on first access
    XPUDriver.utils = _MockUtils()

    print(f"[mock_gpu] Layer 2: XPUDriver mocked (target={_mock_target})")

    # ---- Layer 3: Redirect device='xpu' → device='cpu' ----

    def _redirect_device(device):
        """Convert xpu device references to cpu."""
        if device is None:
            return device
        if isinstance(device, torch.device):
            if device.type == "xpu":
                return torch.device("cpu")
        elif isinstance(device, str):
            if "xpu" in device:
                return device.replace("xpu", "cpu")
        return device

    # Patch tensor factory functions
    _factory_names = (
        "randn", "zeros", "ones", "empty", "full", "rand", "arange",
        "tensor", "randn_like", "zeros_like", "ones_like", "empty_like",
        "randint", "randint_like",
    )
    for fn_name in _factory_names:
        orig = getattr(torch, fn_name, None)
        if orig is None:
            continue

        def _make_wrapper(orig_fn):
            @functools.wraps(orig_fn)
            def wrapper(*args, **kwargs):
                if "device" in kwargs:
                    kwargs["device"] = _redirect_device(kwargs["device"])
                return orig_fn(*args, **kwargs)
            return wrapper

        setattr(torch, fn_name, _make_wrapper(orig))

    # Patch Tensor.to() — handles tensor.to('xpu') and tensor.to(device='xpu')
    _orig_to = torch.Tensor.to

    def _mock_to(self, *args, **kwargs):
        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                args = (_redirect_device(first),) + args[1:]
        if "device" in kwargs:
            kwargs["device"] = _redirect_device(kwargs["device"])
        return _orig_to(self, *args, **kwargs)

    torch.Tensor.to = _mock_to

    # Patch Tensor.xpu() — returns self (tensor stays on cpu)
    torch.Tensor.xpu = lambda self, *a, **kw: self

    print("[mock_gpu] Layer 3: tensor factories redirect xpu → cpu")

    # ---- Layer 4: torch.compile + TorchInductor (FlexAttention) support ----
    #
    # FlexAttention uses torch.compile → TorchDynamo → TorchInductor → Triton.
    # Three problems arise on a GPU-less machine:
    #
    # Problem A: TorchDynamo's SymbolicStreamState.__init__ calls
    #   torch.accelerator.current_stream() → torch._C._accelerator_getDeviceIndex()
    #   → fails with "RuntimeError: No XPU devices are available."
    #
    # Problem B: Input tensors are on CPU (Layer 3 redirects xpu→cpu).
    #   TorchInductor selects backend by device type: CPU → C++ codegen.
    #   We need Triton codegen to trigger ocloc compilation and produce dumps.
    #
    # Problem C: FlexAttention's lowering has its own CPU check:
    #   `if query.get_device().type == "cpu": return lower_cpu(...)`
    #   which fails because check_cpu_supported() returns False when
    #   torch.xpu.is_available() is True (our Layer 1 mock).
    #   Even with cpu_backend="triton", FlexAttention never reaches the
    #   Triton template path for CPU device.
    #
    # Fix A: Mock torch.accelerator functions.
    # Fix B: Set torch._inductor.config.cpu_backend = "triton".
    # Fix C: Override FlexAttention lowering to skip CPU device check.
    #   Also redirect _empty_strided_xpu to CPU so the generated wrapper
    #   code can allocate output tensors without a real XPU device.

    # Fix A: torch.accelerator mocks
    import torch.accelerator

    class _MockAccelStream:
        """Minimal stream mock for torch.accelerator."""
        device = torch.device("cpu")
        sycl_queue = 0

    _accel_stream = _MockAccelStream()
    torch.accelerator.is_available = lambda: True
    torch.accelerator.current_device_index = lambda: 0
    torch.accelerator.device_count = lambda: 1
    torch.accelerator.current_stream = lambda *a, **kw: _accel_stream
    torch.accelerator.set_stream = lambda *a, **kw: None

    # Fix B: Force Triton backend for CPU
    import torch._inductor.config
    torch._inductor.config.cpu_backend = "triton"

    # Fix C: Override FlexAttention lowering to use Triton (GPU) path
    # The lowering checks query.get_device().type == "cpu" and routes to
    # lower_cpu() which fails. We override query.get_device() to return
    # xpu:0 so it enters the Triton template path instead. The generated
    # Triton kernels are compiled by ocloc for the target arch (CRI).
    from torch._inductor.lowering import lowerings as _lowerings
    _flex_hop = torch.ops.higher_order.flex_attention
    if _flex_hop in _lowerings:
        _orig_flex_lower = _lowerings[_flex_hop]

        def _xpu_flex_lower(query, key, value, subgraph, block_mask, scale,
                            kernel_options, score_mod_other_buffers,
                            mask_mod_other_buffers):
            _saved = query.get_device
            query.get_device = lambda: torch.device("xpu:0")
            try:
                return _orig_flex_lower(
                    query, key, value, subgraph, block_mask, scale,
                    kernel_options, score_mod_other_buffers,
                    mask_mod_other_buffers)
            finally:
                query.get_device = _saved

        _lowerings[_flex_hop] = _xpu_flex_lower

    # Same override for the backward lowering.
    _flex_bwd_hop = torch.ops.higher_order.flex_attention_backward
    if _flex_bwd_hop in _lowerings:
        _orig_flex_bwd_lower = _lowerings[_flex_bwd_hop]

        def _xpu_flex_bwd_lower(*args, **kwargs):
            query = args[0]
            _saved = query.get_device
            query.get_device = lambda: torch.device("xpu:0")
            try:
                return _orig_flex_bwd_lower(*args, **kwargs)
            finally:
                query.get_device = _saved

        _lowerings[_flex_bwd_hop] = _xpu_flex_bwd_lower

    # Patch _validate_device in flex_attention eager module. This function
    # raises "FlexAttention does not support backward on CPU" when tensors
    # have requires_grad=True on CPU. It runs before torch.compile, so it
    # can't be caught by the compile_only stub.
    try:
        import torch.nn.attention.flex_attention as _flex_eager_mod
        _flex_eager_mod._validate_device = lambda *a, **kw: None
    except (ImportError, AttributeError):
        pass

    # Redirect _empty_strided_xpu to CPU — the generated TorchInductor
    # wrapper uses this C function to allocate output tensors on XPU.
    # Without override, it segfaults (no SYCL device).
    torch._C._dynamo.guards._empty_strided_xpu = (
        lambda size, stride, dtype: torch.empty_strided(size, stride, dtype=dtype)
    )

    # Override Triton compilation target: TorchInductor creates a
    # GPUTarget('cpu', ...) for CPU tensors. No Triton CPU backend
    # exists, so compilation fails. We redirect to the mocked XPU
    # target (CRI) so the Intel Triton backend + ocloc is used.
    import triton
    _orig_triton_compile = triton.compile

    def _xpu_triton_compile(*args, **kwargs):
        if "target" in kwargs and getattr(kwargs["target"], "backend", None) == "cpu":
            kwargs["target"] = triton.runtime.driver.active.get_current_target()
        return _orig_triton_compile(*args, **kwargs)

    triton.compile = _xpu_triton_compile

    # Force tt.divisibility=16 on every tensor kernel arg. TorchInductor's
    # is_aligned fails for flex_attention's Q/K/V because their offsets are
    # symbolic (GQA head / batch slicing), so statically_known_multiple_of
    # returns False and `divisible_by_16` omits them. Without that hint
    # MaterializeBlockPointer rejects the load (axisInfo.getDivisibility %4
    # check) and Triton falls back to scalar lsc_load — the kernels we want
    # to inspect end up looking 10× larger than their real-GPU equivalents.
    # Monkey-patch is_aligned so every TensorArg becomes divisible_by_16;
    # the real driver enforces 16B-aligned allocations anyway, so this is a
    # safe over-approximation for compile-only dumps.
    try:
        from torch._inductor.codegen import triton_utils as _ti
        from torch._inductor.codegen.common import TensorArg
        _orig_config_of = _ti.config_of

        def _config_of_force_aligned(args, *, indices=None, **kwargs):
            if indices is None:
                indices = list(range(len(args)))
            original = _orig_config_of(args, indices=indices, **kwargs)
            # Triton-2025 AttrsDescriptorWrapper returns a plain dict
            # {(i,): [['tt.divisibility', 16], ...], ...}. Merge in every
            # TensorArg index so flex_attention's Q/K/V get the hint.
            if not isinstance(original, dict):
                return original
            merged = dict(original)
            for i, a in zip(indices, args):
                if isinstance(a, TensorArg) and (i,) not in merged:
                    merged[(i,)] = [["tt.divisibility", 16]]
            return merged

        # Patch the function at its definition site AND every module that
        # did `from .triton_utils import config_of`, since those bound the
        # original reference at import time. select_algorithm.py is the
        # critical one for templates (flex_attention, gemm).
        _ti.config_of = _config_of_force_aligned
        for _mod_name in (
            "torch._inductor.codegen.triton",
            "torch._inductor.codegen.wrapper",
            "torch._inductor.codegen.triton_combo_kernel",
            "torch._inductor.select_algorithm",
        ):
            try:
                import importlib
                _mod = importlib.import_module(_mod_name)
                if hasattr(_mod, "config_of"):
                    _mod.config_of = _config_of_force_aligned
            except ImportError:
                pass
    except (ImportError, AttributeError):
        pass

    # Force can_use_tma()=True so flex_attention emits the TMA template branch.
    # can_use_tma's _is_tma_compatible_matrix rejects a buffer when its name is
    # in V.graph.unaligned_buffers. Layer 3 redirects flex inputs to CPU via
    # torch.empty_strided; the CPU allocator's alignment guarantees don't meet
    # Inductor's TMA rules, so all three Q/K/V inputs land in unaligned_buffers
    # and can_use_tma returns False. flex_attention.py then flips USE_TMA to
    # False and the jinja template emits scalar-pointer loads (tl.load) in
    # place of tl.make_tensor_descriptor + descriptor_load. Real XPU inputs
    # are aligned and stay on the TMA path, so cross-compile ends up shipping
    # a structurally different kernel to IGC — different kernel hash,
    # different instCount, different load_block2d count. Override at the
    # imported-name sites since flex_attention/flex_decoding bound the symbol
    # at module load.
    try:
        import torch._inductor.utils as _iu
        import torch._inductor.kernel.flex.flex_attention as _fa_mod
        import torch._inductor.kernel.flex.flex_decoding as _fd_mod
        _always_true_tma = lambda *a, **kw: True
        _iu.can_use_tma = _always_true_tma
        _fa_mod.can_use_tma = _always_true_tma
        _fd_mod.can_use_tma = _always_true_tma
    except (ImportError, AttributeError):
        pass

    print("[mock_gpu] Layer 4: torch.compile + FlexAttention cross-compile support")


def _filter_triton_only(bench):
    """Patch a Benchmark's line_vals/line_names to keep only triton."""
    triton_idx = [
        i for i, v in enumerate(bench.line_vals)
        if "triton" in str(v).lower()
    ]
    if not triton_idx:
        return
    bench.line_vals = [bench.line_vals[i] for i in triton_idx]
    bench.line_names = [bench.line_names[i] for i in triton_idx]


def _override_x_vals(bench):
    """Override x_vals from AD_TRITON_SIZES env var if set.

    The env var is a Python literal for a single shape:
      Multi-dim: "[1, 1024, 1024, 4096]"
      Single-dim (softmax): "1024"
      With torch dtype: "[1, 1024, 1024, 1024, torch.bfloat16]"

    For multi-dim benchmarks the shape is wrapped in a list (x_vals is a
    list of shapes). For single-dim benchmarks the scalar is used directly.
    """
    ad_sizes = os.environ.get("AD_TRITON_SIZES")
    if not ad_sizes:
        return

    # Use eval() instead of ast.literal_eval() so that torch.bfloat16 etc.
    # can be resolved (needed for gemm_postop_addmatrix benchmarks).
    try:
        parsed = ast.literal_eval(ad_sizes)
    except (ValueError, SyntaxError):
        import torch
        parsed = eval(ad_sizes, {"torch": torch})  # noqa: S307 — trusted input from YAML

    if isinstance(parsed, list):
        bench.x_vals = [parsed]
    else:
        # Single scalar (e.g. softmax N=1024)
        bench.x_vals = [parsed]


def _apply_flex_overrides():
    """Override FlexAttention inductor autotuner configs from env vars.

    AD_OVERRIDE_CONFIGS: Python literal evaluating to a list of FlexConfig.
        Empty string means use defaults (no override).
    AD_FLEX_DECODE_CONFIGS: Python literal evaluating to a list of FlexDecodeConfig.
    AD_BWD_OVERRIDE_CONFIGS: Python literal evaluating to a list of FlexBwDConfig.
    """
    override_str = os.environ.get("AD_OVERRIDE_CONFIGS")  # None if not set
    decode_str = os.environ.get("AD_FLEX_DECODE_CONFIGS")
    bwd_str = os.environ.get("AD_BWD_OVERRIDE_CONFIGS")

    if override_str is None and decode_str is None and bwd_str is None:
        return

    try:
        import torch._inductor.kernel.flex.flex_attention as flex_attn
        from torch._inductor.template_heuristics.triton import FlexConfig
    except ImportError:
        print("[ad_run_benchmark] WARNING: Could not import FlexConfig; skipping config override")
        return

    if override_str:
        configs = eval(override_str)  # noqa: S307  — trusted input from YAML config
        flex_attn.V.choices.get_flex_attention_fwd_configs = lambda *a, **kw: configs

    if decode_str:
        try:
            from torch._inductor.template_heuristics.triton import FlexDecodeConfig
            decode_configs = eval(decode_str)  # noqa: S307
            flex_attn.V.choices.get_flex_decode_configs = lambda *a, **kw: decode_configs
        except ImportError:
            print("[ad_run_benchmark] WARNING: Could not import FlexDecodeConfig; skipping decode config override")

    if bwd_str:
        try:
            from torch._inductor.template_heuristics.triton import FlexBwDConfig
            bwd_configs = eval(bwd_str)  # noqa: S307
            flex_attn.V.choices.get_flex_attn_bwd_configs = lambda *a, **kw: bwd_configs
        except ImportError:
            print("[ad_run_benchmark] WARNING: Could not import FlexBwDConfig; skipping bwd config override")


def _coerce_value(v: str):
    """Convert a string to int/float/bool/None/str."""
    lv = v.strip().lower()
    if lv == "none":
        return None
    if lv == "true":
        return True
    if lv == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v.strip().strip("'\"")


def _force_autotune_config(module):
    """Force a specific Triton @triton.autotune config, bypassing the search.

    Reads AD_TRITON_AUTOTUNE_CONFIG env var: comma-separated KEY=VALUE pairs.
    num_stages and num_warps are top-level Config attributes; all others
    are kernel kwargs.

    Patches every Autotuner object found in the module to contain only a
    single Config matching the specification. If no exact match is found
    among existing configs, creates a new triton.Config.
    """
    config_str = os.environ.get("AD_TRITON_AUTOTUNE_CONFIG")
    if not config_str:
        return

    # Parse the config string
    top_level_keys = {"num_warps", "num_stages", "num_ctas", "maxnreg"}
    kwargs = {}
    top_level = {}
    for pair in config_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        val = _coerce_value(v)
        if k in top_level_keys:
            top_level[k] = val
        else:
            kwargs[k] = val

    try:
        from triton.runtime.autotuner import Autotuner
        import triton
    except ImportError:
        print("[ad_run_benchmark] WARNING: Could not import triton Autotuner; "
              "skipping config override")
        return

    def _match(cfg, effective_kwargs):
        """Check if a Config matches the specification."""
        for k, v in effective_kwargs.items():
            cfg_val = cfg.kwargs.get(k)
            if cfg_val is None:
                return False
            # Compare with type coercion
            if str(cfg_val) != str(v) and cfg_val != v:
                return False
        for k, v in top_level.items():
            cfg_val = getattr(cfg, k, None)
            if cfg_val is not None and cfg_val != v:
                return False
        return True

    def _patch(obj):
        # A benchmark module can expose multiple Autotuners with different
        # kwarg schemas (e.g. flash_attention: _attn_fwd uses BLOCK_M/BLOCK_N
        # while _attn_bwd uses BLOCK_M1/BLOCK_N1/BLOCK_M2/BLOCK_N2). Only
        # force a single Config when the user kwargs fully cover THIS
        # Autotuner's schema; otherwise the forced Config would either
        # drop required kernel kwargs or inject kwargs with the wrong name
        # for the kernel, triggering `dynamic_func() missing ... required
        # positional arguments` at launch.
        existing_keys = set()
        for c in obj.configs:
            existing_keys.update(c.kwargs.keys())
        if existing_keys and not existing_keys.issubset(kwargs.keys()):
            return False
        effective_kwargs = {k: v for k, v in kwargs.items() if k in existing_keys}

        matching = [c for c in obj.configs if _match(c, effective_kwargs)]
        if matching:
            matched = matching[0]
            # Build a new config with ONLY the specified kwargs, using
            # values from the matched config to preserve types (e.g.
            # grf_mode must stay a string '256', not int 256).
            # This avoids extra kwargs in the matched config (like N_CTX
            # in flash attention) that would conflict with benchmark args.
            filtered_kwargs = {k: matched.kwargs.get(k, v) for k, v in effective_kwargs.items()}
            # Merge top-level params: specified override > matched value
            merged_top = {}
            for k in top_level_keys:
                if k in top_level:
                    merged_top[k] = top_level[k]
                elif hasattr(matched, k) and getattr(matched, k) is not None:
                    merged_top[k] = getattr(matched, k)
            obj.configs = [triton.Config(filtered_kwargs, **merged_top)]
        else:
            # Create a new Config with the specified params
            new_cfg = triton.Config(effective_kwargs, **top_level)
            obj.configs = [new_cfg]
        return True

    seen = set()
    patched = 0
    skipped = 0
    # Module-level scan (handles @triton.autotune decorators).
    for attr_name in dir(module):
        obj = getattr(module, attr_name, None)
        if isinstance(obj, Autotuner) and id(obj) not in seen:
            seen.add(id(obj))
            if _patch(obj):
                patched += 1
            else:
                skipped += 1
        elif obj is not None and not isinstance(obj, Autotuner):
            # Deep scan: look one level into classes/objects for nested
            # Autotuners (e.g. flash_attention's _attention.tune_attn_fwd).
            # Skip built-in / stdlib objects by only scanning objects whose
            # module matches the benchmark module to avoid expensive dir()
            # on third-party objects.
            try:
                obj_mod = getattr(obj, "__module__", None)
            except Exception:
                obj_mod = None
            if obj_mod != module.__name__:
                continue
            try:
                nested_names = dir(obj)
            except Exception:
                continue
            for nested_name in nested_names:
                if nested_name.startswith("__"):
                    continue
                try:
                    nested = getattr(obj, nested_name, None)
                except Exception:
                    continue
                if isinstance(nested, Autotuner) and id(nested) not in seen:
                    seen.add(id(nested))
                    if _patch(nested):
                        patched += 1
                    else:
                        skipped += 1

    if patched or skipped:
        skipped_note = f", skipped {skipped} (schema mismatch)" if skipped else ""
        print(f"[ad_run_benchmark] Forced autotune config on {patched} kernel(s){skipped_note}: "
              f"kwargs={kwargs}, {top_level}")
    else:
        print("[ad_run_benchmark] WARNING: No Autotuner objects found in module")


def _patch_compile_only():
    """Replace do_bench with a stub that calls fn() once for compilation.

    In compile-only mode, we want to trigger kernel compilation (generating
    Triton and IGC dumps) without running the full measurement loop.  The
    stub calls fn() exactly once — enough to trigger torch.compile JIT and
    the ocloc compilation that produces IGC dumps for the target device.

    When cross-compiling (TRITON_INTEL_DEVICE_ARCH is set), additional
    patches are applied:
    - XPULauncher.__call__ is patched to a no-op so the kernel is never
      dispatched to the GPU.
    - XPUUtils.load_binary is patched to a no-op so L0 zeModuleCreate is
      never called (no native-device recompilation, no native dumps).
    - assert_close is patched to a no-op (verification is meaningless for
      cross-compiled kernels).
    The JIT pipeline (including ocloc) still runs fully during compilation,
    producing all target-device dumps.  IGC_ShaderDumpEnable stays set
    since L0's IGC is never invoked.
    """
    import triton_kernels_benchmark as _bm

    _cross_compile = bool(os.environ.get("TRITON_INTEL_DEVICE_ARCH"))
    if _cross_compile:
        # Patch XPULauncher.__call__ to skip GPU dispatch.
        from triton.backends.intel.driver import XPULauncher
        XPULauncher.__call__ = lambda self, *args: None

        # Patch _init_handles to skip L0 module creation (load_binary +
        # shared memory checks).  Cannot patch load_binary directly — it's
        # an instance attribute on XPUUtils, and instantiating XPUUtils
        # triggers L0 device init which segfaults with ProductFamilyOverride
        # pointing to non-native hardware.  _init_handles is the call site
        # for load_binary; replacing it avoids all L0 interaction.
        from triton.compiler.compiler import CompiledKernel
        def _noop_init_handles(self):
            if self.module is not None:
                return
            self.module = "dummy"
            self.function = None
            self.n_regs = self.n_spills = self.n_max_threads = 0
        CompiledKernel._init_handles = _noop_init_handles

        # Patch assert_close to skip verification — cross-compiled kernels
        # can't run on the native device anyway.
        _bm.assert_close = lambda *a, **kw: None

        print("[compile_only] Cross-compile mode: patched XPULauncher + "
              "_init_handles (no GPU dispatch, no L0 module creation)")

    # Use a tiny non-zero time (1e-3 ms) to avoid division-by-zero in
    # benchmark functions that compute GB/s = bytes / (ms * 1e-3).
    _DUMMY_MS = 1e-3

    def _make_compile_stub(quantiles=None, **_kw):
        """Create a do_bench replacement that knows the expected return shape."""
        def _stub(fn, *_args, **_kwargs):
            try:
                fn()
            except Exception as e:
                print(f"[compile_only] Execution error (may be expected): {e}")
            if quantiles is not None:
                # quantile values + mean + cv
                return [_DUMMY_MS] * len(quantiles) + [_DUMMY_MS, 0.0]
            return _DUMMY_MS
        return _stub

    # Patch get_do_bench so new partials use the stub
    _bm.get_do_bench = lambda *a, **kw: _make_compile_stub(**kw)

    # Patch do_bench directly for callers that don't go through get_do_bench
    _bm.do_bench = _make_compile_stub(quantiles=[0.5, 0.0, 1.0])

    mode = "cross-compile" if _cross_compile else "native"
    print(f"[compile_only] Patched do_bench — kernels will compile but not benchmark ({mode})")


def _filter_masks(bench):
    """Filter x_vals to only include specified mask types.

    AD_FLEX_MASKS: comma-separated mask names, e.g. "NATTEN,Alibi".
    The mask name is expected in one of the x_val elements (string).
    """
    masks_str = os.environ.get("AD_FLEX_MASKS")
    if not masks_str:
        return

    allowed = {m.strip() for m in masks_str.split(",")}
    filtered = []
    for shape in bench.x_vals:
        # Find the mask element: it's a string in MASKS
        for elem in shape:
            if isinstance(elem, str) and elem in (
                "NATTEN", "Alibi", "Noop", "Softcap", "PagedNoop",
                "fwd", "bwd",
            ):
                if elem in allowed:
                    filtered.append(shape)
                    break
                elif elem in ("fwd", "bwd"):
                    # MODE field, not a mask — keep checking
                    continue
                else:
                    break
    if filtered:
        bench.x_vals = filtered


def _fixup_flash_attention_forward(module):
    """Patch _attention.forward to match the current kernel signature.

    The checked-in flash_attention_benchmark.py call site was never updated
    after the kernel migrated to tensor descriptors (commit 5fee15d3c) —
    it still passes 24 positional args + N_CTX kwarg, which matches the
    old `_attn_fwd_with_block_pointers(Q, K, V, sm_scale, M, Out, strides...)`
    signature but collides with the new `_attn_fwd(sm_scale, M, Z, H, Q, K, V, O,
    N_CTX, ...)` signature (N_CTX ends up filled by q.stride(2) positionally
    AND by the N_CTX kwarg → "multiple values for N_CTX").

    We cannot edit the benchmark file (it ships via the pip package), so we
    replace `_attention.forward` with a staticmethod that issues the correct
    call for the default (non-advanced) path. This only touches the autotuned
    forward path; the backward path (tune_attn_bwd) is unchanged.
    """
    if module.__name__ != "triton_kernels_benchmark.flash_attention_benchmark":
        return
    import torch
    import triton
    _attention_cls = getattr(module, "_attention", None)
    if _attention_cls is None:
        return

    @staticmethod
    def _patched_forward(ctx, q, k, v, causal, sm_scale):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        grid = lambda args: (q.shape[0], q.shape[1], triton.cdiv(q.shape[2], args['BLOCK_M']))
        n_ctx = q.shape[2]
        if n_ctx <= 512:
            grid = lambda args: (triton.cdiv(q.shape[2], args['BLOCK_M']), 1, q.shape[0] * q.shape[1])
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        _attention_cls.tune_attn_fwd[grid](  # pylint: disable=unsubscriptable-object
            sm_scale, M,
            q.shape[0], q.shape[1],
            q, k, v, o,
            N_CTX=q.shape[2],
            HEAD_DIM=Lk,
            STAGE=stage,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = Lk
        ctx.causal = causal
        return o

    _attention_cls.forward = _patched_forward
    # _attention.apply is a torch.autograd.Function descriptor; reassign so
    # the module-level `attention` alias picks up the new forward.
    module.attention = _attention_cls.apply
    print("[ad_run_benchmark] Patched flash_attention _attention.forward "
          "to match current kernel signature")


def _fixup_flex_sdpa_reference(module):
    """Skip the sycl-tla SDPA reference comparison in cross-compile mode.

    flex_attention_benchmark_causal_mask.benchmark() runs an SDPA reference
    (via F.scaled_dot_product_attention under sdpa_kernel(FLASH_ATTENTION))
    before the Triton kernel whenever D_HEAD_qk == D_HEAD_v, so it can
    assert_close triton vs sycl-tla. With AD_MOCK_GPU=1 there is no XPU
    device, so the reference dispatch routes to FlashAttentionXPU which
    raises 'XPU device is not available' — aborting the x_vals loop in
    Mark._run before the Triton kernel is ever compiled. Result: zero IGC
    dumps for the 96 decode/append flex_bwd configs (shapes where
    D_HEAD_qk == D_HEAD_v).

    Replace get_sdpa_benchmark with a stub that:
    - Returns None for the fwd path — the benchmark already handles
      `if sycl_fn is not None: assert_close(...)`.
    - Returns a grad-connected fake (bwd_fn, do, o) tuple for the bwd path
      so the downstream `_, _, sycl_o = sdpa_result` unpack and
      `torch.autograd.grad((sycl_o,), (q, k, v), ...)` call succeed
      without touching FlashAttentionXPU. assert_close is already patched
      to a no-op by _patch_compile_only, so the fake tensor values are
      irrelevant — only the grad graph shape needs to be valid.
    """
    if os.environ.get("AD_MOCK_GPU") != "1":
        return
    if not module.__name__.startswith("triton_kernels_benchmark.flex_attention_benchmark"):
        return
    if not hasattr(module, "get_sdpa_benchmark"):
        return

    import torch

    def _fake_sdpa(q, k, v, attn_bias, use_causal, sm_scale, H_q, H_kv,
                   D_HEAD_qk, D_HEAD_v, MODE, provider, backwards_grad=None):
        # Preserve the original early-return so D_HEAD_qk != D_HEAD_v still
        # takes the "reference not available" branch.
        if D_HEAD_qk != D_HEAD_v or (provider == 'onednn' and MODE == 'bwd'):
            return None
        if MODE != 'bwd':
            # fwd path just needs a callable; assert_close is a no-op and
            # the lambda is not invoked in cross-compile mode.
            return lambda: q.sum() + k.sum() + v.sum()
        # bwd path: need (bwd_fn, do, sycl_o) where sycl_o has grad
        # connectivity to (q, k, v) so torch.autograd.grad succeeds.
        Z, N_q = q.shape[0], q.shape[2]
        scalar = q.sum() + k.sum() + v.sum()
        fake_o = scalar * torch.ones(
            (Z, H_q, N_q, D_HEAD_v), device=q.device, dtype=q.dtype)
        do = backwards_grad if backwards_grad is not None else torch.randn_like(fake_o)
        bwd_fn = lambda: fake_o.backward(do, retain_graph=True)
        return bwd_fn, do, fake_o

    module.get_sdpa_benchmark = _fake_sdpa
    print("[ad_run_benchmark] Patched flex_attention get_sdpa_benchmark "
          "to skip SDPA reference in cross-compile mode")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <benchmark_key>")
        print(f"Available benchmarks: {', '.join(sorted(BENCHMARKS))}")
        sys.exit(1)

    key = sys.argv[1]
    if key not in BENCHMARKS:
        print(f"Unknown benchmark: {key}")
        print(f"Available: {', '.join(sorted(BENCHMARKS))}")
        sys.exit(1)

    spec = BENCHMARKS[key]

    # 1. Mock native extensions before any benchmark import
    _mock_native_modules()

    # 1.5. Mock GPU subsystem for cross-compilation on GPU-less machines
    if os.environ.get("AD_MOCK_GPU") == "1":
        _mock_gpu_for_cross_compile()

    # 2. Set env vars before import (some modules read them at import time)
    for k, v in spec.get("env_before_import", {}).items():
        os.environ.setdefault(k, v)

    # 3. Strip our args from sys.argv so the benchmark framework's argparse
    #    doesn't choke on them. Keep only the script name.
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]

    # 4. Import the benchmark module and get the Mark object
    module = importlib.import_module(f"triton_kernels_benchmark.{spec['module']}")

    # 5. Force specific autotune config if requested. Called twice:
    #    - Before get_benchmark() for kernels whose Autotuner is created by
    #      @triton.autotune at module import (most benchmarks).
    #    - After get_benchmark() for kernels whose Autotuner is created
    #      inside get_benchmark() (flash_attention assigns _attention.tune_attn_fwd
    #      = tuner(attn_fwd) only when the factory runs).
    #    _force_autotune_config is idempotent: once a config list is reduced
    #    to one entry, the second pass re-matches and rewrites the same single
    #    entry. Duplicate Autotuner instances are skipped via id() tracking.
    _force_autotune_config(module)

    if "factory" in spec:
        kwargs = {**spec.get("kwargs", {}), "providers_filter": ["triton"]}
        mark = getattr(module, spec["factory"])(**kwargs)
    else:
        mark = getattr(module, spec["attr"])

    # Flash attention's checked-in _attention.forward uses an obsolete call
    # pattern that does not match the current kernel; rewrite it before the
    # benchmark runs so both native and cross-compile paths produce dumps.
    _fixup_flash_attention_forward(module)

    # Skip sycl-tla SDPA reference when mocking the GPU — otherwise flex_bwd
    # decode/append shapes abort before Triton codegen runs.
    _fixup_flex_sdpa_reference(module)

    _force_autotune_config(module)

    bench = mark.benchmarks

    # 6. Filter to triton provider only (for non-factory modules)
    if "attr" in spec:
        _filter_triton_only(bench)

    # 7. Override x_vals from AD_TRITON_SIZES
    _override_x_vals(bench)

    # 8. Override FlexAttention configs
    if spec.get("flex_attention"):
        _apply_flex_overrides()

    # 9. Filter masks for custom masks benchmark
    if os.environ.get("AD_FLEX_MASKS"):
        _filter_masks(bench)

    # 10. Compile-only mode: replace do_bench with single-call stub
    if os.environ.get("AD_COMPILE_ONLY") == "1":
        _patch_compile_only()

    # 11. Run the benchmark
    # In compile_only mode, some benchmarks (e.g. flex_bwd) call the compiled
    # function eagerly outside do_bench to compute reference outputs. Those
    # calls are not covered by the do_bench stub, so exceptions propagate and
    # crash the process. Catch them here so the run exits cleanly.
    if os.environ.get("AD_COMPILE_ONLY") == "1":
        try:
            mark.run(show_plots=False, print_data=True)
        except Exception as e:
            print(f"[compile_only] Benchmark runner error (expected in compile-only mode): "
                  f"{type(e).__name__}: {e}")
    else:
        mark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()

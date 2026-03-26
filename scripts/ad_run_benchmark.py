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

    def _match(cfg):
        """Check if a Config matches the specification."""
        for k, v in kwargs.items():
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

    patched = 0
    for attr_name in dir(module):
        obj = getattr(module, attr_name, None)
        if isinstance(obj, Autotuner):
            matching = [c for c in obj.configs if _match(c)]
            if matching:
                obj.configs = matching[:1]
            else:
                # Create a new Config with the specified params
                new_cfg = triton.Config(kwargs, **top_level)
                obj.configs = [new_cfg]
            patched += 1

    if patched:
        print(f"[ad_run_benchmark] Forced autotune config on {patched} kernel(s): "
              f"kwargs={kwargs}, {top_level}")
    else:
        print("[ad_run_benchmark] WARNING: No Autotuner objects found in module")


def _patch_compile_only():
    """Replace do_bench with a stub that calls fn() once for compilation.

    In compile-only mode, we want to trigger kernel compilation (generating
    Triton and IGC dumps) without running the full measurement loop.  The
    stub calls fn() exactly once — enough to trigger torch.compile JIT and
    IGC shader compilation — then returns zero-valued stats.
    """
    import triton_kernels_benchmark as _bm

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

    print("[compile_only] Patched do_bench — kernels will compile but not benchmark")


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

    # 2. Set env vars before import (some modules read them at import time)
    for k, v in spec.get("env_before_import", {}).items():
        os.environ.setdefault(k, v)

    # 3. Strip our args from sys.argv so the benchmark framework's argparse
    #    doesn't choke on them. Keep only the script name.
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]

    # 4. Import the benchmark module and get the Mark object
    module = importlib.import_module(f"triton_kernels_benchmark.{spec['module']}")

    # 5. Force specific autotune config if requested (before get_benchmark
    #    which may trigger compilation)
    _force_autotune_config(module)

    if "factory" in spec:
        kwargs = {**spec.get("kwargs", {}), "providers_filter": ["triton"]}
        mark = getattr(module, spec["factory"])(**kwargs)
    else:
        mark = getattr(module, spec["attr"])

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
    mark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()

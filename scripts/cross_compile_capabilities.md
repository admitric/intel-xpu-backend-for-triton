# Cross-compile capability flags — how to verify against production

The `AD_MOCK_GPU=1` cross-compile path in [`scripts/ad_run_benchmark.py`](ad_run_benchmark.py) hardcodes a `_ARCH_CAPABILITIES` table that maps the target arch string (`bmg`, `pvc`, `cri`, `ptl_h`, `ptl_u`) to `has_*` capability flags. These flags gate `MaterializeBlockPointer`, `Pipeline`, `AccelerateMatmul`/`DPASAnalysis` in the Triton Intel XPU backend. If they drift from what a real Level Zero / SYCL driver reports, cross-compile dumps stop matching real on-GPU compilations and become unreliable for performance analysis.

This doc explains how to verify the table, what to compare, and what to do when it drifts.

## TL;DR

On a machine with a real Intel GPU (BMG, PVC, CRI, …), run the snippet at the bottom to dump the `has_*` dict for every relevant capability, then diff it against the matching arch entry in `_ARCH_CAPABILITIES`.

## The capability surface

The mock must cover the full set of keys Triton reads in [`third_party/intel/backend/compiler.py`](../third_party/intel/backend/compiler.py) `parse_target()` / `annotate_module()`:

| Capability key | Gates | Source on real hardware |
|----------------|-------|-------------------------|
| `has_2d_block_io` | `MaterializeBlockPointer`, `Pipeline`, block 2D loads/stores | `cl_intel_subgroup_2d_block_io` OpenCL extension |
| `has_subgroup_matrix_multiply_accumulate` | `AccelerateMatmul` / DPAS lowering | `cl_intel_subgroup_matrix_multiply_accumulate` |
| `has_subgroup_matrix_multiply_accumulate_tensor_float32` | TF32 DPAS variant | `cl_intel_subgroup_matrix_multiply_accumulate_tensor_float32` |
| `has_subgroup_matrix_multiply_accumulate_bfloat8` | BF8/HF8 DPAS (Xe3P+) | Capability query, Xe3P+ only |
| `has_subgroup_scaled_matrix_multiply_accumulate` | Block-scale DPAS (BDPAS) | Capability query |
| `has_bfloat16_conversion` | BF16 conversion ops | `cl_intel_bfloat16_conversions` |
| `has_bfloat16_arithmetic` | Native BF16 arithmetic | Driver version gate (`is_lts`) + device info |
| `has_predicated_io` | Predicated load/store | Driver version gate + device info |
| `has_f8_conversions` | FP8 (E5M2/E4M3FN) conversions | Device info |
| `has_f4_conversions` | FP4 (E2M1) conversions | Device info |
| `has_16bit_atomics` | Native 16-bit atomics | Device info |
| `has_256b_prefetch` | 256-byte 2D block prefetch | Device info |
| `has_fp64` | Double-precision support | Device info |

Only the first four are auto-populated by [`query_device_extensions`](../third_party/intel/backend/extension_utils.py). The rest come from the driver-populated `target_props` dict on real hardware. The mock has to simulate ALL of them.

## How to dump the ground truth on a real machine

Run this on a machine with the target GPU (BMG / PVC / CRI):

```bash
ssh <target-machine>
conda activate triton
source <gfx-driver-build>/source_me   # driver + NEO

python - <<'PY'
import torch
from triton.backends.intel.driver import XPUDriver
from triton.backends.intel import extension_utils

# 1. Snapshot the target dict that compiler.parse_target() would see
tgt = XPUDriver().get_current_target()
print("arch string:", tgt.arch.get("arch", "?"))
print("raw target arch dict:")
for k in sorted(tgt.arch):
    print(f"  {k!r}: {tgt.arch[k]!r}")

# 2. Compute the derived dev_prop dict exactly as compiler.parse_target() does
tgt_prop = tgt.arch
dev_prop = {}
for key, default in [
    ("has_2d_block_io", False),
    ("has_subgroup_matrix_multiply_accumulate", False),
    ("has_subgroup_matrix_multiply_accumulate_bfloat8", False),
    ("has_subgroup_matrix_multiply_accumulate_tensor_float32", False),
    ("has_subgroup_scaled_matrix_multiply_accumulate", False),
    ("has_bfloat16_conversion", False),
    ("has_bfloat16_arithmetic", False),   # is_lts may change default; see note
    ("has_predicated_io", False),         # is_lts may change default
    ("has_f8_conversions", False),
    ("has_f4_conversions", False),
    ("has_16bit_atomics", False),
    ("has_256b_prefetch", False),
    ("has_fp64", None),
]:
    dev_prop[key] = tgt_prop.get(key, default)
print("\ndev_prop (post-parse_target, pre-annotate_module):")
for k, v in dev_prop.items():
    print(f"  {k}: {v}")

# 3. If the tgt_prop above was missing extension keys (no __intel_already_queried_extensions__
#    sentinel), also query directly — this is what compiler.py falls back to.
device_id = extension_utils.get_device_id(0)
print(f"\ndevice_id from SYCL: {device_id}")
print("query_device_extensions(device_id):")
for k, v in extension_utils.query_device_extensions(device_id).items():
    print(f"  {k}: {v}")
PY
```

**Note:** `has_bfloat16_arithmetic` and `has_predicated_io` default to `not is_lts`. On a release driver they are typically True; on an LTS driver they can be False. The mock hardcodes True for these (matching release-driver behavior). If your target run uses an LTS driver, override via `TRITON_INTEL_DEVICE_EXTENSIONS` env var or clear them in `_ARCH_CAPABILITIES`.

## Cross-checking the mock

1. On the target machine, run the dump script above and save the `dev_prop` block.
2. On the cross-compile host, run the mock and dump the resolved target:
   ```bash
   AD_MOCK_GPU=1 TRITON_INTEL_DEVICE_ARCH=<arch> python - <<'PY'
   from scripts.ad_run_benchmark import _mock_gpu_for_cross_compile
   _mock_gpu_for_cross_compile()
   from triton.backends.intel.driver import XPUDriver
   tgt = XPUDriver().get_current_target()
   for k in sorted(tgt.arch):
       print(f"  {k!r}: {tgt.arch[k]!r}")
   PY
   ```
3. Diff the two outputs on the `has_*` keys. Any mismatch in the first two columns of the table above is a **functional** difference (fast-path selection will diverge from production). The remaining ones are second-order but still matter for fidelity.

An easier end-to-end check: compile the same kernel (same SPIR-V) under each setup with `IGC_ShaderDumpEnable=1` and compare `codegen.ll` for:
- Counts of `@llvm.genx.GenISA.LSC2DBlockRead`
- Counts of `@llvm.genx.GenISA.LSC2DBlockWrite`
- Counts of `@llvm.genx.GenISA.sub_group_dpas`

If any of these differ by more than a handful between cross-compile and real-GPU compile, the capability flags are drifting.

## What to do when you find drift

1. **Missing key in `_ARCH_CAPABILITIES`:** add it with the value from the real-GPU dump.
2. **New key in `compiler.py`:** extend the table in `ad_run_benchmark.py` to cover it and mention the new key in the table above.
3. **Arch-specific divergence (e.g., new board with same arch string but different caps):** either add a dedicated arch entry or switch to env-var overrides via `TRITON_INTEL_DEVICE_EXTENSIONS` at run time.

The env var is additive on top of `_ARCH_CAPABILITIES`, so per-run tweaks don't require code changes:
```bash
TRITON_INTEL_DEVICE_EXTENSIONS="has_f4_conversions has_256b_prefetch" \
TRITON_INTEL_DEVICE_ARCH=bmg AD_MOCK_GPU=1 python scripts/ad_run_benchmark.py …
```

## Regression guard

To catch capability drift early:
- `configs/sanity_cross_compile_cri.yaml` runs a small set of representative kernels.
- After the run, grep each `dumps/*_codegen.ll` for `LSC2DBlockRead`/`sub_group_dpas` counts.
- On a healthy run, flex_fwd / gemm / gemm_bt kernels should show **> 0** `LSC2DBlockRead` and non-zero DPAS equivalents (via the vectorizer-driven path). If any of them shows 0, capability flags are probably wrong or a recently-added one is missing from the mock.

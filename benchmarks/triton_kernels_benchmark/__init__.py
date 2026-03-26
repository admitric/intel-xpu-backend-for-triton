import os

from triton_kernels_benchmark.benchmark_testing import (
    assert_close,
    do_bench,
    filter_providers,
    perf_report,
    Benchmark,
    BenchmarkCategory,
    BenchmarkConfig,
    BENCHMARKING_CONFIG,
    BENCHMARKING_METHOD,
    get_do_bench,
    get_total_gpu_memory_bytes,
)

from triton_kernels_benchmark.benchmark_shapes_parser import ShapePatternParser

# Native C++ extensions (xetla, onednn, sycl_tla) are optional — they may
# not be built when TRITON_BENCHMARKS_SKIP_NATIVE=1 is set.  Provide None
# stubs so that `from triton_kernels_benchmark import xetla_kernel` doesn't
# crash at import time.  Benchmark modules should guard usage with
# `if xetla_kernel is not None:`.
try:
    from triton_kernels_benchmark import xetla_kernel
except ImportError:
    xetla_kernel = None  # type: ignore[assignment]

try:
    from triton_kernels_benchmark import onednn_kernel
except ImportError:
    onednn_kernel = None  # type: ignore[assignment]

try:
    from triton_kernels_benchmark import sycl_tla_kernel
except ImportError:
    sycl_tla_kernel = None  # type: ignore[assignment]

if BENCHMARKING_METHOD == "UPSTREAM_PYTORCH_PROFILER":
    os.environ["INJECT_PYTORCH"] = "True"

__all__ = [
    "assert_close",
    "do_bench",
    "filter_providers",
    "perf_report",
    "Benchmark",
    "BenchmarkCategory",
    "BenchmarkConfig",
    "BENCHMARKING_CONFIG",
    "BENCHMARKING_METHOD",
    "ShapePatternParser",
    "get_do_bench",
]

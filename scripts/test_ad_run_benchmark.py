"""Tests for scripts/ad_run_benchmark.py helpers.

Run with:
    conda activate triton && pytest scripts/test_ad_run_benchmark.py -v
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ad_run_benchmark = pytest.importorskip("ad_run_benchmark")
triton = pytest.importorskip("triton")
autotuner_mod = pytest.importorskip("triton.runtime.autotuner")
Autotuner = autotuner_mod.Autotuner


def _make_tuner(configs):
    """Build a minimal Autotuner-typed object carrying just `.configs`.

    _force_autotune_config only reads/writes `.configs`, so we skip the real
    __init__ (which requires a jit-compiled kernel).
    """
    tuner = object.__new__(Autotuner)
    tuner.configs = list(configs)
    return tuner


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("AD_TRITON_AUTOTUNE_CONFIG", raising=False)
    return monkeypatch


def test_force_autotune_config_skips_autotuner_with_mismatched_schema(clean_env):
    """Regression: flash_bwd has two Autotuners with different kwarg schemas.

    _attn_fwd's Autotuner uses BLOCK_M/BLOCK_N. _attn_bwd's Autotuner uses
    BLOCK_M1/BLOCK_N1/BLOCK_M2/BLOCK_N2. Applying backward kwargs to both
    produced a forward Config missing BLOCK_M/BLOCK_N, which then failed as
    `TypeError: dynamic_func() missing 2 required positional arguments`.

    After the fix: the forward Autotuner must be left alone (kwargs that
    don't match its schema should not force a replacement config).
    """
    fwd_tuner = _make_tuner([
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "grf_mode": "auto"},
                      num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "grf_mode": "auto"},
                      num_warps=8, num_stages=3),
    ])
    bwd_tuner = _make_tuner([
        triton.Config({"BLOCK_M1": 64, "BLOCK_N1": 64,
                       "BLOCK_M2": 64, "BLOCK_N2": 64, "grf_mode": "auto"},
                      num_warps=4, num_stages=2),
    ])

    mod = types.ModuleType("fake_flash_bench")
    mod.fwd_tuner = fwd_tuner
    mod.bwd_tuner = bwd_tuner

    clean_env.setenv(
        "AD_TRITON_AUTOTUNE_CONFIG",
        "BLOCK_M1=32,BLOCK_N1=128,BLOCK_M2=128,BLOCK_N2=32,"
        "grf_mode=256,num_stages=2,num_warps=16",
    )

    ad_run_benchmark._force_autotune_config(mod)

    # Forward Autotuner: schema not fully covered by user kwargs, so the
    # whole config list must be left untouched. Previously the code built
    # a single Config with backward-shaped kwargs, which crashed at launch
    # as `dynamic_func() missing ... required positional arguments` because
    # _attn_fwd expects BLOCK_M/BLOCK_N, not BLOCK_M1/BLOCK_N1.
    assert len(fwd_tuner.configs) == 2, (
        f"forward Autotuner should be left alone; got {fwd_tuner.configs}"
    )
    for cfg in fwd_tuner.configs:
        assert "BLOCK_M" in cfg.kwargs, (
            f"forward config lost BLOCK_M after patching: {cfg.kwargs}"
        )
        assert "BLOCK_N" in cfg.kwargs
        assert "BLOCK_M1" not in cfg.kwargs, (
            f"forward config gained backward-only kwarg BLOCK_M1: {cfg.kwargs}"
        )

    # Backward Autotuner: user kwargs cover its full schema, so force a
    # single matching Config.
    assert len(bwd_tuner.configs) == 1
    only = bwd_tuner.configs[0]
    assert only.kwargs["BLOCK_M1"] == 32
    assert only.kwargs["BLOCK_N1"] == 128
    assert only.kwargs["BLOCK_M2"] == 128
    assert only.kwargs["BLOCK_N2"] == 32
    assert str(only.kwargs["grf_mode"]) == "256"
    assert only.num_warps == 16
    assert only.num_stages == 2


def test_force_autotune_config_applies_when_schema_fully_covered(clean_env):
    """Sanity: the existing flash_fwd path keeps working.

    For a single Autotuner whose schema is fully covered by user kwargs,
    the behavior must match the pre-fix behavior: collapse to one Config.
    """
    fwd_tuner = _make_tuner([
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "grf_mode": "auto"},
                      num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "grf_mode": "auto"},
                      num_warps=8, num_stages=3),
    ])

    mod = types.ModuleType("fake_fwd_only_bench")
    mod.fwd_tuner = fwd_tuner

    clean_env.setenv(
        "AD_TRITON_AUTOTUNE_CONFIG",
        "BLOCK_M=64,BLOCK_N=64,grf_mode=256,num_stages=2,num_warps=16",
    )

    ad_run_benchmark._force_autotune_config(mod)

    assert len(fwd_tuner.configs) == 1
    only = fwd_tuner.configs[0]
    assert only.kwargs["BLOCK_M"] == 64
    assert only.kwargs["BLOCK_N"] == 64
    assert str(only.kwargs["grf_mode"]) == "256"
    assert only.num_warps == 16
    assert only.num_stages == 2

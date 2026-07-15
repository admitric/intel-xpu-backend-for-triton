// Cost gate for the last-tile mask peel. The same peelable K-loop is compiled
// three ways to exercise both env knobs:
//   - default threshold (128): body is small, so the loop IS peeled (trailing scf.if);
//   - TRITON_INTEL_MASK_PEEL_MAX_BODY_OPS=1: body exceeds the cap, so NOT peeled;
//   - TRITON_INTEL_DISABLE_MASK_PEEL=1: peel disabled outright, so NOT peeled.
// In the NOPEEL cases no other RemoveMasks strategy fires either (the bound is a
// runtime value and the mask is loop-variant), so the masked load stays in place.

// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s --check-prefix=PEEL
// RUN: TRITON_INTEL_MASK_PEEL_MAX_BODY_OPS=1 triton-opt %s -triton-intel-remove-masks | FileCheck %s --check-prefix=NOPEEL
// RUN: TRITON_INTEL_DISABLE_MASK_PEEL=1 triton-opt %s -triton-intel-remove-masks | FileCheck %s --check-prefix=NOPEEL

module {
  tt.func public @peel_cost_gate(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                 %N: i32 {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %other = arith.constant dense<0> : tensor<32xi32>
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %Nsplat = tt.splat %N : i32 -> tensor<32xi32>
    %res = scf.for %iv = %c0 to %N step %c32 iter_args(%acc = %other) -> (tensor<32xi32>) : i32 {
      %ivs = tt.splat %iv : i32 -> tensor<32xi32>
      %off = arith.addi %ivs, %r : tensor<32xi32>
      %mask = arith.cmpi slt, %off, %Nsplat : tensor<32xi32>
      %ptrs = tt.addptr %base, %off : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
      %val = tt.load %ptrs, %mask, %other : tensor<32x!tt.ptr<i32>>
      %new = arith.addi %acc, %val : tensor<32xi32>
      scf.yield %new : tensor<32xi32>
    }
    %ptrs2 = tt.addptr %base, %r : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    tt.store %ptrs2, %res : tensor<32x!tt.ptr<i32>>
    tt.return
  }

  // Default threshold: peeled -> the mask moves into a trailing scf.if.
  // PEEL-LABEL: @peel_cost_gate
  // PEEL:      scf.for
  // PEEL:      scf.if
  // PEEL:        tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i32>>

  // Gated off (either knob): no peel, the masked load stays inside the single loop.
  // NOPEEL-LABEL: @peel_cost_gate
  // NOPEEL-NOT: scf.if
  // NOPEEL:     scf.for
  // NOPEEL:       tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i32>>
  // NOPEEL-NOT: scf.if
}

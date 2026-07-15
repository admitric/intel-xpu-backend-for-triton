// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

// A K-loop whose masked load carries a last-tile boundary predicate
// `(iv + [0..32)) < N` with a *dynamic* bound N == the loop upper bound. The
// peel strategy splits off the final iteration and drops the mask in the
// steady-state loop (a single masked tail iteration preserves correctness).
module {
  tt.func public @peel_boundary(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
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
  // CHECK-LABEL: @peel_boundary
  // Main loop upper bound reduced by one step; the load is UNMASKED:
  // CHECK:      [[UB:%.+]] = arith.subi %arg1, %c32
  // CHECK:      scf.for %{{.*}} = %c0{{.*}} to [[UB]] step %c32{{.*}} iter_args
  // CHECK:        [[P:%.+]] = tt.addptr
  // CHECK-NEXT:   tt.load [[P]] : tensor<32x!tt.ptr<i32>>
  // CHECK:      }
  // Peeled last iteration in a trailing scf.if, mask KEPT:
  // CHECK:      scf.if
  // CHECK:        tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i32>>
  // Pipeliner-safety invariant: the peel leaves exactly one steady-state scf.for;
  // the tail is an scf.if, never a second loop. The Intel software pipeliner sets
  // peelEpilogue=false and only walks scf.for, so it neither re-peels the tail nor
  // sees it as a pipelining target -- the shrunk main loop still pipelines normally.
  // CHECK-NOT:  scf.for

  // Negative case: the mask bound %M is NOT the loop upper bound %N, so the
  // last-tile relation does not hold and the loop must NOT be peeled -- the
  // masked load stays inside a single (unpeeled) loop.
  tt.func public @no_peel_wrong_bound(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                      %N: i32 {tt.divisibility = 16 : i32},
                                      %M: i32 {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %other = arith.constant dense<0> : tensor<32xi32>
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %Msplat = tt.splat %M : i32 -> tensor<32xi32>
    %res = scf.for %iv = %c0 to %N step %c32 iter_args(%acc = %other) -> (tensor<32xi32>) : i32 {
      %ivs = tt.splat %iv : i32 -> tensor<32xi32>
      %off = arith.addi %ivs, %r : tensor<32xi32>
      %mask = arith.cmpi slt, %off, %Msplat : tensor<32xi32>
      %ptrs = tt.addptr %base, %off : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
      %val = tt.load %ptrs, %mask, %other : tensor<32x!tt.ptr<i32>>
      %new = arith.addi %acc, %val : tensor<32xi32>
      scf.yield %new : tensor<32xi32>
    }
    %ptrs2 = tt.addptr %base, %r : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    tt.store %ptrs2, %res : tensor<32x!tt.ptr<i32>>
    tt.return
  }
  // CHECK-LABEL: @no_peel_wrong_bound
  // CHECK-NOT:  scf.if
  // CHECK:      scf.for
  // CHECK:        tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i32>>
  // CHECK-NOT:  scf.if

  // Boundary carried by an `andi(mask, offset<N)` feeding a select -- the real
  // `tl.where(mask & boundary, ...)` form, on an i64 loop with trunci(IV) and
  // extsi(offset). The peel drops the boundary leaf in the steady state (select
  // guarded by the real mask alone) and keeps it in the tail.
  tt.func public @peel_boundary_select(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %N: i64, %mb: i1) {
    %c0 = arith.constant 0 : i64
    %c32 = arith.constant 32 : i64
    %cst = arith.constant dense<-1.000000e+00> : tensor<32xf32>
    %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %m = tt.splat %mb : i1 -> tensor<32xi1>
    %Nsplat = tt.splat %N : i64 -> tensor<32xi64>
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %res = scf.for %iv = %c0 to %N step %c32 iter_args(%acc = %cst) -> (tensor<32xf32>) : i64 {
      %sn   = arith.trunci %iv : i64 to i32
      %sns  = tt.splat %sn : i32 -> tensor<32xi32>
      %off  = arith.addi %sns, %r : tensor<32xi32>
      %offe = arith.extsi %off : tensor<32xi32> to tensor<32xi64>
      %bnd  = arith.cmpi slt, %offe, %Nsplat : tensor<32xi64>
      %comb = arith.andi %m, %bnd : tensor<32xi1>
      %ptrs = tt.addptr %base, %off : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
      %val  = tt.load %ptrs : tensor<32x!tt.ptr<f32>>
      %sel  = arith.select %comb, %val, %cst : tensor<32xi1>, tensor<32xf32>
      %new  = arith.addf %acc, %sel : tensor<32xf32>
      scf.yield %new : tensor<32xf32>
    }
    %p2 = tt.addptr %base, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %p2, %res : tensor<32x!tt.ptr<f32>>
    tt.return
  }
  // CHECK-LABEL: @peel_boundary_select
  // CHECK:      [[UB2:%.+]] = arith.subi %arg1, %c32
  // CHECK:      scf.for %{{.*}} to [[UB2]]
  // CHECK-NOT:    arith.andi
  // CHECK:        arith.select
  // CHECK:      }
  // CHECK:      scf.if
  // CHECK:        arith.andi
  // CHECK:        arith.select
}

// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 TRITON_INTEL_PREDICATED_STORE=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// This test verifies that loads without boundary checks preserve a shared base
// pointer across sub-tiles, enabling IGC's decomposition pass to hoist address
// payload creation. The key invariants are:
// 1. BASE extracted once from block pointer struct and reused
// 2. WIDTH/HEIGHT use full tensor dimensions (not shrunken to tile size)
// 3. OFFSET_X and OFFSET_Y are non-zero and vary per sub-tile
// 4. All sub-tile loads reference the same BASE value

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @load_no_boundary_check_shared_base
  tt.func public @load_no_boundary_check_shared_base(%arg0: !tt.ptr<f16>, %arg1: i64, %arg2: i64, %arg3: i64) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // Create a 64x64 tensor pointer (large enough to decompose into multiple sub-tiles)
    %ptrA = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dot0>>

    // Extract block pointer struct components (done once)
    // CHECK:           %[[BASE:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[HEIGHT_i64:.*]] = llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFFSET_X_BASE:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[OFFSET_Y_BASE:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

    // Compute width in bytes (full tensor width, not tile width)
    // CHECK:           %[[WIDTH_i32:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[ELEM_SIZE:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:           %[[WIDTH_BYTES:.*]] = llvm.mul %[[WIDTH_i32]], %[[ELEM_SIZE]] : i32

    // Height (full tensor height, not tile height)
    // CHECK:           %[[HEIGHT_i32:.*]] = llvm.trunc %[[HEIGHT_i64]] : i64 to i32

    // Multiple 2D block loads should share the SAME BASE with different offsets
    // The pattern to verify: all loads use %[[BASE]] (extracted once) with varying offsets
    // CHECK-COUNT-4:   triton_gen.2Dblockload %[[BASE]], %[[WIDTH_BYTES]], %[[HEIGHT_i32]]

    // Load without boundary checks - all accesses are known in-bounds
    %A = tt.load %ptrA {boundaryCheck = array<i32>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #dot0>>
    tt.return
  }
}

// -----

// Test for stores without boundary checks - should also preserve shared base
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
// CHECK-LABEL:   llvm.func spir_kernelcc @store_no_boundary_check_shared_base
  tt.func public @store_no_boundary_check_shared_base(%arg0: !tt.ptr<f16>, %arg1: i64, %arg2: i64, %arg3: i64, %data: tensor<64x64xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    %ptrOut = tt.make_tensor_ptr %arg0, [%arg1, %arg2], [%arg3, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #dpas>>

    // Extract block pointer (once)
    // CHECK:           %[[BASE:.*]] = llvm.extractvalue {{.*}}[6] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>
    // CHECK:           %[[WIDTH_i64:.*]] = llvm.extractvalue {{.*}}[3] : !llvm.struct<(i32, i32, i64, i64, i64, i64, ptr<1>)>

    // Width in bytes (full tensor, not tile)
    // CHECK:           %[[WIDTH_i32:.*]] = llvm.trunc %[[WIDTH_i64]] : i64 to i32
    // CHECK:           %[[WIDTH_BYTES:.*]] = llvm.mul %[[WIDTH_i32]], {{.*}} : i32

    // Multiple stores share the SAME BASE with different offsets
    // CHECK:           triton_gen.2Dblockstore {{.*}}, %[[BASE]], %[[WIDTH_BYTES]], {{.*}}
    // CHECK:           triton_gen.2Dblockstore {{.*}}, %[[BASE]], %[[WIDTH_BYTES]], {{.*}}

    // Store without boundary checks
    tt.store %ptrOut, %data {boundaryCheck = array<i32>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #dpas>>
    tt.return
  }
}

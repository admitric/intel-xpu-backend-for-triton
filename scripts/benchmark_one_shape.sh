#!/bin/bash
bench_name=$1
full_bench_name=benchmarks/triton_kernels_benchmark/$bench_name
sed -i "s/x_vals=.*/x_vals=[$AD_TRITON_SIZES],/g" $full_bench_name
python $full_bench_name


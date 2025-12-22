#!/bin/bash
bench_name=$1
full_bench_name=benchmarks/triton_kernels_benchmark/$bench_name

# Check if AD_TRITON_SIZES_EXTENDED is set, and split it if needed
if [ -n "$AD_TRITON_SIZES_EXTENDED" ]; then
    # Split by | to get AD_TRITON_SIZES and AD_OVERRIDE_CONFIGS
    AD_TRITON_SIZES=$(echo "$AD_TRITON_SIZES_EXTENDED" | cut -d'|' -f1)
    AD_OVERRIDE_CONFIGS=$(echo "$AD_TRITON_SIZES_EXTENDED" | cut -d'|' -f2)
fi

sed -i "s/x_vals=.*/x_vals=[$AD_TRITON_SIZES],/g" $full_bench_name
sed -i "s/override_configs = .*/override_configs = ${AD_OVERRIDE_CONFIGS:-configs}/" "$full_bench_name"
python $full_bench_name


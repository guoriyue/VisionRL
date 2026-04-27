#!/bin/bash
# GPU utilization sampler — writes CSV with ~100ms granularity.
# Usage: bash gpu_monitor.sh <out.csv>
OUT="${1:-gpu_util.csv}"
echo "epoch_ms,util_gpu,util_mem,mem_used_mib" > "$OUT"
START_MS=$(date +%s%3N)
while true; do
    # nvidia-smi dmon -c 1 is the fastest single sample
    line=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used \
                     --format=csv,noheader,nounits | head -1)
    NOW_MS=$(date +%s%3N)
    ELAPSED=$((NOW_MS - START_MS))
    # normalize: "12, 5, 3456" -> "12,5,3456"
    clean=$(echo "$line" | tr -d ' ')
    echo "${ELAPSED},${clean}" >> "$OUT"
    sleep 0.1
done

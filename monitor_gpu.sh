#!/bin/bash

# Remove the previous log file
rm -f detailed_gpu_log.csv

# Command to log NVIDIA GPU stats
counter=0
while [ $counter -lt 200 ]; do
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> detailed_gpu_log.csv
    sleep 1
    ((counter++))
done
#!/bin/bash

# 设置检查间隔（秒）
CHECK_INTERVAL=300

# 设置 GPU 空闲阈值（单位：MB，可根据实际情况调整）
MEMORY_THRESHOLD=100  # 显存占用低于 100MB 视为空闲
UTILIZATION_THRESHOLD=5  # GPU 利用率低于 5% 视为空闲

# 日志文件路径
LOG_FILE="gpu_monitor.log"

# 检查 GPU 是否空闲的函数
check_gpu_idle() {
    local gpu_id=$1
    local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    local utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)

    if [ "$memory_used" -lt "$MEMORY_THRESHOLD" ] && [ "$utilization" -lt "$UTILIZATION_THRESHOLD" ]; then
        return 0  # GPU 空闲
    else
        return 1  # GPU 忙碌
    fi
}

# 主循环
while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] Checking GPU status..." >> "$LOG_FILE"

    if check_gpu_idle 0 && check_gpu_idle 1; then
        echo "[$TIMESTAMP] GPU0 and GPU1 are idle. Starting training task..." >> "$LOG_FILE"

        # 执行训练任务
        accelerate launch train.py --config configs/resnet34_cifar100_base.yaml >> "$LOG_FILE" 2>&1

        # 任务完成后退出监控（如需持续监控，可删除此行）
        echo "[$TIMESTAMP] Training task completed." >> "$LOG_FILE"
        break
    else
        echo "[$TIMESTAMP] GPU0 or GPU1 is busy. Waiting..." >> "$LOG_FILE"
    fi

    sleep $CHECK_INTERVAL
done
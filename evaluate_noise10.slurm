#!/bin/bash
#SBATCH --job-name=evaluate             # 作业名称
#SBATCH --output=evaluate.out           # 标准输出文件
#SBATCH --error=evaluate.err            # 标准错误文件
#SBATCH --partition=GPU-small,GPU-shared # 分区名称（根据集群配置进行调整）
#SBATCH --gres=gpu:v100:1               # 请求1个V100 GPU
#SBATCH --time=08:00:00                 # 作业运行时间限制（8小时）
#SBATCH --mem=12G                       # 内存限制（12GB）


# 运行Python脚本
python yucheng/evaluate_noise10.py

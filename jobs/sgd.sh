#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --partition=partition_name
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o output.log
#SBATCH --ntasks=1

conda activate env_name
cd experiments


WD=0

Cs=(1.0)
LRs=(0.5 0.2 0.1 0.05 0.02 0.01)
BSs=(8 16 32 64)

num_c=${#Cs[@]}
num_bs=${#BSs[@]}
num_lr=${#LRs[@]}

total_combinations=$((num_lr * num_bs * num_c))

if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
    exit 1
fi

lr_index=$((SLURM_ARRAY_TASK_ID / (num_bs * num_c)))
bs_index=$(((SLURM_ARRAY_TASK_ID / num_c) % num_bs))

LR=${LRs[$lr_index]}
BS=${BSs[$bs_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-sgd-LR${LR}
mkdir -p "../logs/$TIME"

python sgd.py --seed 0 \
            --epochs 100 \
            --private_bs $BS \
            --dataset cifar10 \
            --num_test_per_epoch 4 \
            --pretrain nfresnet18 \
            --lr $LR \
            --wd $WD \
            --time "$TIME" 2>&1 |tee "../logs/$TIME/log.out"
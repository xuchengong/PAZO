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

LR=0.1
BS=8
CLIPNORM=0.1

# Cs=(0.01 0.1 0.5 1.0 2.0)
# LRs=(0.005)
# BSs=(8 32 128)

# num_cs=${#Cs[@]}
# num_bs=${#BSs[@]}
# num_lr=${#LRs[@]}

# total_combinations=$((num_lr * num_bs * num_cs))

# if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
#     echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
#     exit 1
# fi

# lr_index=$((SLURM_ARRAY_TASK_ID / (num_bs * num_cs)))
# bs_index=$(((SLURM_ARRAY_TASK_ID / num_cs) % num_bs))
# cs_index=$((SLURM_ARRAY_TASK_ID % num_cs))

# LR=${LRs[$lr_index]}
# BS=${BSs[$bs_index]}
# CLIPNORM=${Cs[$cs_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-dpmd-C${CLIPNORM}-LR${LR}
mkdir -p "../logs/$TIME"

python dpmd.py --epochs 100 \
                --dataset imdb10k \
                --num_microbatches 8 \
                --public_bs $BS \
                --pretrain lstm \
                --sigma 1.25 \
                --private_bs 64 \
                --eval_every_epoch 2 \
                --lr $LR \
                --clipping_bound $CLIPNORM \
                --seed 0 \
                --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
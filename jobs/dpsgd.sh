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

LR=2.0
CLIPNORM=0.1

# Cs=(2.0)
# LRs=(0.005)

# num_c=${#Cs[@]}
# num_lr=${#LRs[@]}

# total_combinations=$((num_lr * num_c))
# if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
#     echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
#     exit 1
# fi

# c_index=$(((SLURM_ARRAY_TASK_ID / num_lr) % num_c))
# lr_index=$((SLURM_ARRAY_TASK_ID % num_lr))

# LR=${LRs[$lr_index]}
# CLIPNORM=${Cs[$c_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-dpsgd-C${CLIPNORM}-LR${LR}
mkdir -p "../logs/$TIME"

python dpsgd_vec.py --lr $LR \
                    --public_size 2000 \
                    --pretrain nfresnet18 \
                    --dataset cifar10 \
                    --eval_every_epoch 2 \
                    --epochs 100 \
                    --sigma 1.25 \
                    --private_bs 64 \
                    --num_microbatches 1 \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"

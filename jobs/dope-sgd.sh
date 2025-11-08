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

# Cs=(0.1 0.5 1.0 2.0 4.0)
# LRs=(0.001)
# BSs=(8 32)

# num_c=${#Cs[@]}
# num_bs=${#BSs[@]}
# num_lr=${#LRs[@]}

# total_combinations=$((num_lr * num_bs * num_c))

# if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
#     echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
#     exit 1
# fi

# lr_index=$((SLURM_ARRAY_TASK_ID / (num_bs * num_c)))
# bs_index=$(((SLURM_ARRAY_TASK_ID / num_c) % num_bs))
# c_index=$((SLURM_ARRAY_TASK_ID % num_c))

# LR=${LRs[$lr_index]}
# BS=${BSs[$bs_index]}
# CLIPNORM=${Cs[$c_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-dope-BS${BS}-C${CLIPNORM}-LR${LR}
mkdir -p "../logs/$TIME"

python dope-sgd_vec.py --epochs 100 \
                    --dataset cifar10 \
                    --public_size 2000 \
                    --num_microbatches 2 \
                    --public_bs $BS \
                    --pretrain nfresnet18 \
                    --eval_every_epoch 2 \
                    --lr $LR \
                    --sigma 1.25 \
                    --private_bs 64 \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"

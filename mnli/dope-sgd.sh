#!/bin/bash

#SBATCH --job-name=job_name
#SBATCH --partition=partition_name
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o output.log
#SBATCH --ntasks=1

conda activate env_name
cd mnli

LR=5e-5
BS=32
CLIPNORM=50.0

# Cs=(10.0 20.0 50.0)
# LRs=(2e-4)
# BSs=(8)

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

python dope-sgd.py --epochs 100 \
                    --dataset mnli_snli_512  \
                    --public_bs $BS \
                    --pretrain roberta \
                    --eval_every_epoch 2 \
                    --num_microbatches 32 \
                    --lr $LR \
                    --sigma 6.2 \
                    --private_bs 64 \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
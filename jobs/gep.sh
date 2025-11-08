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


CLIPNORM1=1.0
BS=32
LR=0.01

# Cs=(0.1)
# LRs=(0.05 0.02 0.01)
# BSs=(8 32)

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
# CLIPNORM1=${Cs[$cs_index]}

CLIPNORM0=$(echo "$CLIPNORM1 * 5" | bc)
TIME=$(date +'%y-%m-%d-%H-%M-%S')-gep-BS${BS}-C${CLIPNORM1}-LR${LR}
mkdir -p "../logs/$TIME"

python gep_vec.py --epochs 100 \
                    --public_size 2000 \
                    --public_bs $BS \
                    --num_microbatches 2 \
                    --pretrain nfresnet18 \
                    --dataset cifar10 \
                    --eval_every_epoch 2 \
                    --lr $LR \
                    --num_groups 3 \
                    --private_bs 64 \
                    --target_eps 1.25 \
                    --power_iter 1 \
                    --clip0 $CLIPNORM0 \
                    --clip1 $CLIPNORM1 \
                    --seed 0 \
                    --num_bases $BS \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
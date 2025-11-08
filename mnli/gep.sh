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


CLIP=-1
SEED=0

CLIPNORM1=1.0
BS=32
LR=0.1

# Cs=(20.0 50.0 100.0 150.0 200.0 250.0 300.0)
# LRs=(2e-6 5e-6 1e-5 2e-5 5e-5 1e-4 2e-4)
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
TIME=$(date +'%y-%m-%d-%H-%M-%S')-gep-BS${BS}-C${CLIPNORM1}-LR${LR}-GEP
mkdir -p "../logs/$TIME"

python gep.py --epochs 100 \
                --public_bs $BS \
                --pretrain roberta \
                --dataset mnli_snli_512 \
                --eval_every_epoch 2 \
                --num_microbatches 8 \
                --lr $LR \
                --num_groups 1 \
                --private_bs 64 \
                --target_eps 3.0 \
                --power_iter 1 \
                --clip0 $CLIPNORM0 \
                --clip1 $CLIPNORM1 \
                --seed 0 \
                --num_bases $BS \
                --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
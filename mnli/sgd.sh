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

WD=0

Cs=(1.0)
LRs=(5e-3 1e-3 1e-4 1e-5 1e-6)
BSs=(8 32 64)

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
            --dataset mnli_snli_512 \
            --num_test_per_epoch 4 \
            --pretrain roberta \
            --lr $LR \
            --wd $WD \
            --time "$TIME" 2>&1 |tee "../logs/$TIME/log.out"
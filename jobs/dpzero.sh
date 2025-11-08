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

CLIPNORM=1.0

ND=1
LR=0.1

# NDs=(1 5)
# LRs=(0.005 0.01 0.02 0.05)

# num_lr=${#LRs[@]}
# num_nd=${#NDs[@]}

# total_combinations=$((num_lr * num_nd))
# if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
#     echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
#     exit 1
# fi

# nd_index=$(((SLURM_ARRAY_TASK_ID / num_lr) % num_nd))
# lr_index=$((SLURM_ARRAY_TASK_ID % num_lr))

# LR=${LRs[$lr_index]}
# ND=${NDs[$nd_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-0th-C${CLIPNORM}-LR${LR}
mkdir -p "../logs/$TIME"

python dpzero.py --epochs 100 \
                --public_size 2000 \
                --perturbation_scale 0.01 \
                --pretrain nfresnet18 \
                --dataset cifar10 \
                --num_directions $ND \
                --eval_every_epoch 2 \
                --lr $LR \
                --sigma 1.25 \
                --private_bs 64 \
                --clipping_bound $CLIPNORM \
                --seed 0 \
                --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
                    
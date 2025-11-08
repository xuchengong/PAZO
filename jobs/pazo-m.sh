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

BS=8
PER=0.01

ND=1
COE=0.5
LR=1.0

# CLIPNORMs=(0.1 0.5 1.0 2.0)
# COEs=(0.5)
# LRs=(0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001)
# NDs=(1 5)

# num_cs=${#CLIPNORMs[@]}
# num_coe=${#COEs[@]}
# num_lr=${#LRs[@]}
# num_nd=${#NDs[@]}

# total_combinations=$((num_lr * num_coe * num_nd * num_cs))

# if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
#     echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
#     exit 1
# fi

# c_index=$((SLURM_ARRAY_TASK_ID / (num_coe * num_nd * num_lr)))
# coe_index=$((SLURM_ARRAY_TASK_ID / (num_nd * num_lr) % num_coe))
# lr_index=$(((SLURM_ARRAY_TASK_ID / num_nd) % num_lr))
# nd_index=$((SLURM_ARRAY_TASK_ID % num_nd))

# CLIPNORM=${CLIPNORMs[$c_index]}
# ND=${NDs[$nd_index]}
# COE=${COEs[$coe_index]}
# LR=${LRs[$lr_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-m-C${CLIPNORM}-LR${LR}-BS${BS}
mkdir -p "../logs/$TIME"

python pazo-m.py --epochs 100 \
                    --public_size 2000 \
                    --public_bs $BS \
                    --perturbation_scale $PER \
                    --pretrain nfresnet18 \
                    --dataset cifar10 \
                    --coefficient $COE \
                    --num_directions $ND \
                    --eval_every_epoch 2 \
                    --lr $LR \
                    --sigma 2.1 \
                    --private_bs 64 \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"

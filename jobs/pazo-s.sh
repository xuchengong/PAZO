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

NC=3

# LR=4.0
# BS=8
# EPS=0.001
# CLIPNORM=1.0


EPSs=(0.01)
LRs=(0.02 0.01 0.005 0.001 0.0005 0.0002 0.0001)
BSs=(8 32 128)
CLIPNORMs=(0.5 1.0 2.0 4.0)

num_eps=${#EPSs[@]}
num_bs=${#BSs[@]}
num_lr=${#LRs[@]}
num_cs=${#CLIPNORMs[@]}

total_combinations=$((num_lr * num_bs * num_eps * num_cs))

if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
    exit 1
fi

c_index=$((SLURM_ARRAY_TASK_ID / (num_bs * num_eps * num_lr)))
lr_index=$(((SLURM_ARRAY_TASK_ID / (num_bs * num_eps)) % num_lr))
bs_index=$(((SLURM_ARRAY_TASK_ID / num_eps) % num_bs))
eps_index=$((SLURM_ARRAY_TASK_ID % num_eps))

LR=${LRs[$lr_index]}
BS=${BSs[$bs_index]}
EPS=${EPSs[$eps_index]}
CLIPNORM=${CLIPNORMs[$c_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-s-BS${BS}x${NC}-C${CLIPNORM}-LR${LR}-EPS${EPS}
mkdir -p "../logs/$TIME"

python pazo-s.py --epochs 100 \
                    --public_bs $BS \
                    --pretrain vit \
                    --dataset tiny-imagenet \
                    --num_candidate $NC \
                    --eval_every_epoch 2 \
                    --lr $LR \
                    --sigma 1.25 \
                    --private_bs 64 \
                    --epsilon_scale $EPS \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
                    
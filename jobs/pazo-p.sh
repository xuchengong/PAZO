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

ND=5
PER=0.01

# CLIPNORM=1.0
# NC=3
# LR=0.1 # 0.2 0.1 0.05 0.01
# BS=16

CLIPNORMs=(0.5 1.0 2.0)
NCs=(3 6 10)
LRs=(1.0 0.5 0.2 0.1 0.05)
BSs=(32 16 8)

num_cs=${#CLIPNORMs[@]}
num_nc=${#NCs[@]}
num_bs=${#BSs[@]}
num_lr=${#LRs[@]}

total_combinations=$((num_lr * num_bs * num_nc * num_cs))

if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
    exit 1
fi

c_index=$((SLURM_ARRAY_TASK_ID / (num_bs * num_nc * num_lr)))
lr_index=$((SLURM_ARRAY_TASK_ID / (num_bs * num_nc) % num_lr))
bs_index=$(((SLURM_ARRAY_TASK_ID / num_nc) % num_bs))
nc_index=$((SLURM_ARRAY_TASK_ID % num_nc))

CLIPNORM=${CLIPNORMs[$c_index]}
LR=${LRs[$lr_index]}
BS=${BSs[$bs_index]}
NC=${NCs[$nc_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-p-BS${BS}x${NC}-C${CLIPNORM}-LR${LR}
mkdir -p "../logs/$TIME"

python pazo-p.py --epochs 100 \
                    --public_bs $BS \
                    --num_candidate $NC \
                    --perturbation_scale $PER \
                    --pretrain vit \
                    --dataset tiny-imagenet \
                    --sigma 1.25 \
                    --private_bs 64 \
                    --num_directions $ND \
                    --eval_every_epoch 2 \
                    --lr $LR \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"     
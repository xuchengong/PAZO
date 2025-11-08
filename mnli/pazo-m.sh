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

BS=8

# CLIPNORM=100.0
# ND=1
# LR=1e-4
# COE=0.5


CLIPNORMs=(50.0 100.0 150.0 200.0 250.0 300.0 400.0)
COEs=(0.25 0.75)
LRs=(1e-4 2e-4 5e-4 1e-3 2e-3)
NDs=(1)

num_cs=${#CLIPNORMs[@]}
num_coe=${#COEs[@]}
num_lr=${#LRs[@]}
num_nd=${#NDs[@]}

total_combinations=$((num_lr * num_coe * num_nd * num_cs))

if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID is out of range. Exiting."
    exit 1
fi

c_index=$((SLURM_ARRAY_TASK_ID / (num_coe * num_nd * num_lr)))
coe_index=$((SLURM_ARRAY_TASK_ID / (num_nd * num_lr) % num_coe))
lr_index=$(((SLURM_ARRAY_TASK_ID / num_nd) % num_lr))
nd_index=$((SLURM_ARRAY_TASK_ID % num_nd))

CLIPNORM=${CLIPNORMs[$c_index]}
ND=${NDs[$nd_index]}
LR=${LRs[$lr_index]}
COE=${COEs[$coe_index]}

TIME=$(date +'%y-%m-%d-%H-%M-%S')-m-C${CLIPNORM}-LR${LR}-BS${BS}
mkdir -p "../logs/$TIME"

python pazo-m.py --epochs 100 \
                    --public_bs $BS \
                    --perturbation_scale 0.001 \
                    --pretrain roberta \
                    --dataset mnli_snli_512 \
                    --coefficient $COE \
                    --num_directions $ND \
                    --eval_every_epoch 2 \
                    --lr $LR \
                    --sigma 6.2 \
                    --private_bs 64 \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"

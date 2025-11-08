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

PER=0.001

CLIPNORM=100.0
ND=1
LR=5e-6 

CLIPNORMs=(10.0)
COEs=(0.5)
LRs=(1e-4 2e-4 5e-4)
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


TIME=$(date +'%y-%m-%d-%H-%M-%S')-dpzero-C${CLIPNORM}-LR${LR}-ND${ND}-BS${BS}
mkdir -p "../logs/$TIME"

python dpzero.py --epochs 200 \
                    --perturbation_scale $PER \
                    --pretrain roberta \
                    --dataset mnli_snli_512 \
                    --num_directions $ND \
                    --eval_every_epoch 2 \
                    --lr $LR \
                    --sigma 3.49 \
                    --private_bs 64 \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
                    # --continue_from $TIME \

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


ND=5
PER=0.001

# CLIPNORM=50.0
# NC=3
# LR=1e-3 # 0.2 0.1 0.05 0.01
# BS=32

CLIPNORMs=(50.0 100.0 150.0 200.0 250.0 300.0 400.0)
NCs=(10 6 3)
LRs=(2e-3 1e-3 5e-4 2e-4 1e-4 5e-5 2e-5 1e-5)
BSs=(128)

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
                    --pretrain roberta \
                    --dataset mnli_snli_512 \
                    --sigma 6.2 \
                    --private_bs 64 \
                    --num_directions $ND \
                    --eval_every_epoch 2 \
                    --num_microbatches 5 \
                    --lr $LR \
                    --clipping_bound $CLIPNORM \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"     
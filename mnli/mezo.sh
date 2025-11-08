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

SEED=0
ND=5
PER=0.001


for BS in 16 64
do
    for LR in 2e-6 5e-6
    do
        TIME=$(date +'%y-%m-%d-%H-%M-%S')-mezo-BS${BS}-PER${PER}
        mkdir -p "../logs/$TIME"

        python mezo.py --epochs 100 \
                    --perturbation_scale $PER \
                    --pretrain roberta \
                    --dataset mnli \
                    --private_bs $BS \
                    --num_directions $ND \
                    --eval_every_epoch 2 \
                    --lr $LR \
                    --seed 0 \
                    --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
                            
    done
done

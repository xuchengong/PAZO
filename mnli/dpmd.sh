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

# BS=32
for BS in 8
do
    for CLIPNORM in 10.0
    do
        for LR in 1e-5
        do
            TIME=$(date +'%y-%m-%d-%H-%M-%S')-dpmd-C${CLIPNORM}-LR${LR}
            mkdir -p "../logs/$TIME"

            python dpmd.py --epochs 100 \
                            --dataset mnli_snli_512 \
                            --public_bs $BS \
                            --pretrain roberta \
                            --sigma 44.9 \
                            --private_bs 64 \
                            --num_microbatches 64 \
                            --eval_every_epoch 2 \
                            --lr $LR \
                            --clipping_bound $CLIPNORM \
                            --seed 0 \
                            --time "$TIME" 2>&1 |tee -a "../logs/$TIME/log.out"
        done
    done
done
#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

num_layers_lst=(2 8 16 32 64)
alpha_lst=(0.1 0.2 0.5)
theta_lst=(0.5 1 1.5)

for num_layers in "${num_layers_lst[@]}"; do
    for alpha in "${alpha_lst[@]}"; do
        for theta in "${theta_lst[@]}"; do
                if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                    python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method gcn2 --lr 0.01 --num_layers $num_layers --hidden_channels 64 --dropout 0.5 --gcn2_alpha $alpha --theta $theta --display_step 25 --runs 5 --directed
                else
                    python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method gcn2 --lr 0.01 --num_layers $num_layers --hidden_channels 64  --dropout 0.5 --gcn2_alpha $alpha --theta $theta --display_step 25 --runs 5
                fi
        done
    done
done

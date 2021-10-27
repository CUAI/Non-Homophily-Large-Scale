#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=(16 32 128 256)
num_layers_lst=(1 2 3)


for num_layers in "${num_layers_lst[@]}"; do
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
            echo "Running $dataset "
            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx  --num_layers $num_layers --hidden_channels $hidden_channels --display_step 25 --runs 5 --directed
        else
            python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx  --num_layers $num_layers --hidden_channels $hidden_channels --display_step 25 --runs 5
        fi
    done
done

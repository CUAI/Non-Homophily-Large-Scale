#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

weight_decay_lst=(.001 .01 .1)
hidden_channels_lst=(16 32 64 128 256)
num_layers_lst=(2 3)

for weight_decay in "${weight_decay_lst[@]}"; do
    for num_layers in "${num_layers_lst[@]}"; do
        for hidden_channels in "${hidden_channels_lst[@]}"; do
            if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                echo "Running $dataset "
                python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method link_concat --weight_decay $weight_decay --num_layers $num_layers --hidden_channels $hidden_channels --display_step 25 --runs 5 --directed
            else
                python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method link_concat --weight_decay $weight_decay --num_layers $num_layers --hidden_channels $hidden_channels --display_step 25 --runs 5
            fi
        done
    done
done

#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=(128)

num_layers_lst=(2 4)

num_parts_lst=(200)
cluster_batch_size_lst=(1 5)


for num_parts in "${num_parts_lst[@]}"; do
    for cluster_batch_size in "${cluster_batch_size_lst[@]}"; do
        for hidden_channels in "${hidden_channels_lst[@]}"; do
            for num_layers in "${num_layers_lst[@]}"; do
                    python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                        --method gcnjk --jk_type cat --num_layers $num_layers --hidden_channels $hidden_channels \
                        --display_step 25 --runs 5 --train_batch cluster --num_parts $num_parts --cluster_batch_size $cluster_batch_size
            done
        done
    done
done

#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=(128)

num_layers_lst=(2 4)

batch_size_lst=(5000 10000)


for batch_size in "${batch_size_lst[@]}"; do
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for num_layers in "${num_layers_lst[@]}"; do
            if [ "$dataset" = "arxiv-year" ] || [ "$dataset" = "genius" ] || ["$dataset" = "fb100"]; then
                python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                    --method gcnjk --jk_type cat --num_layers $num_layers --hidden_channels $hidden_channels \
                    --display_step 25 --runs 5 --train_batch graphsaint-node --batch_size $batch_size \
                    --no_mini_batch_test --saint_num_steps 5
            else

                python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                    --method gcnjk --jk_type cat --num_layers $num_layers --hidden_channels $hidden_channels \
                    --display_step 25 --runs 5 --train_batch graphsaint-node --batch_size $batch_size \
                    --saint_num_steps 5 --test_num_parts 10
            fi
        done
    done
done
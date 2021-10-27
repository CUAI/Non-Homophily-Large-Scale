#!/bin/bash
# tuning script for LINKX on Geom-GCN datasets. 

dataset=$1
sub_dataset=${2:-''}

A_layers_lst=(1 2)
X_layers_lst=(1 2)
hidden_channels_lst=(64 128 256 512)
num_layers_lst=(1 2 3 4)
lr_lst=(0.05 0.01 0.002)
dropout_lst=(0 .5)


for A_layers in "${A_layers_lst[@]}"; do
    for X_layers in "${X_layers_lst[@]}"; do
        for hidden_channels in "${hidden_channels_lst[@]}"; do
            for num_layers in "${num_layers_lst[@]}"; do
                for lr in "${lr_lst[@]}"; do
                    for dropout in "${dropout_lst[@]}"; do
                        python main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method linkx --hidden_channels $hidden_channels --num_layers $num_layers --lr $lr --dropout $dropout --display_step 25 --runs 10 --link_init_layers_A $A_layers --link_init_layers_X $X_layers 
                    done
                done 
            done 
        done 
    done 
done 

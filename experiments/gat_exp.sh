#!/bin/bash

dataset_lst=("fb100" "arxiv-year" "pokec" "snap-patents" "genius" "twitch-gamer")

if [ "$dataset" = "fb100" ]; then
	sub_dataset="Penn94"
else
	sub_dataset="None"
fi

hidden_channels_lst=()
gat_heads_lst=()

if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "pokec" ]; then 
    gat_heads_lst=(2 4)
    hidden_channels_lst=(4 8 12)
else 
    gat_heads_lst=(2 4 8)
    hidden_channels_lst=(4 8 12 32)
fi 

lr_lst=(0.1 0.01 0.001)
for dataset in "${dataset_lst[@]}"; do
	for hidden_channels in "${hidden_channels_lst[@]}"; do
			for gat_heads in "${gat_heads_lst[@]}"; do
					for lr in "${lr_lst[@]}"; do

				if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
								python main.py --dataset $dataset --sub_dataset $sub_dataset \
								--method gat --num_layers 2 --hidden_channels $hidden_channels \
								--lr $lr --gat_heads $gat_heads --directed --display_step 25 --runs 5 
				else
					python main.py --dataset $dataset --sub_dataset $sub_dataset \
								--method gat --num_layers 2 --hidden_channels $hidden_channels \
								--lr $lr --gat_heads $gat_heads --display_step 25 --runs 5 
				fi
done            
done
done
done 

#!/bin/bash

dataset_lst=("fb100" "arxiv-year" "snap-patents" "pokec" "genius" "twitch-gamer") 
sub_dataset="Penn94" # Only fb100 uses sub_dataset

hidden_channels_lst=()

if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "pokec" ]; then 
    hidden_channels_lst=(4 8 16 32)
else 
    hidden_channels_lst=(4 8 16 32 64)
fi 

lr_lst=(0.1 0.01 0.001)

for dataset in "${dataset_lst[@]}"; do
	for hidden_channels in "${hidden_channels_lst[@]}"; do
					for lr in "${lr_lst[@]}"; do
						if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
							python main.py --dataset $dataset --sub_dataset $sub_dataset \
							--method gcn --num_layers 2 --hidden_channels $hidden_channels \
							--lr $lr  --display_step 25 --runs 5  --directed
						else
							python main.py --dataset $dataset --sub_dataset $sub_dataset \
							--method gcn --num_layers 2 --hidden_channels $hidden_channels \
							--lr $lr  --display_step 25 --runs 5 
						fi
done            
done
done 

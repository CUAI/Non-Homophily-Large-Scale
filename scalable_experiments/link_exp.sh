#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

weight_decay_lst=(.001 .01 .1)
for weight_decay in "${weight_decay_lst[@]}"; do
	if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
		echo "Running $dataset "
    	python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method link --weight_decay $weight_decay  --display_step 100 --runs 5 --directed --train_batch row --num_parts 10
	else
    	python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method link --weight_decay $weight_decay  --display_step 100 --runs 5 --train_batch row --num_parts 10
	fi
done

#!/bin/bash
# training command for different datasets.
source_dir=../dataset
emb_dir=../dataset/Biaffine/glove
save_dir=saved_models

exp_setting=train

####### Restaurants  acc:83.55 f1:75.99 #########
exp_dataset=Biaffine/glove/Restaurants
exp_path=$save_dir/Restaurants/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

# END=50
# for ((i=0;i<=END;i++)); 
# 	do 
# 		# CUDA_VISIBLE_DEVICES=0 python -u train.py \
# 		# 	--data_dir $source_dir/$exp_dataset \
# 		# 	--vocab_dir $source_dir/$exp_dataset \
# 		# 	--glove_dir $emb_dir \
# 		# 	--model "RGAT" \
# 		# 	--save_dir $exp_path \
# 		# 	--seed $i \
# 		# 	--pooling "avg" \
# 		# 	--output_merge "None" \
# 		# 	--num_layers 6 \
# 		# 	--attn_heads 10 \
# 		# 	--num_epoch 65 \
# 		# 	--shuffle \
# 		# 	--name $i \
# 		# 	--modify 0 \
# 		# 	--aspect 0 \
# 		# 	2>&1 | tee $exp_path/training.log

# 		CUDA_VISIBLE_DEVICES=0 python -u train.py \
# 			--data_dir $source_dir/$exp_dataset \
# 			--vocab_dir $source_dir/$exp_dataset \
# 			--glove_dir $emb_dir \
# 			--model "RGAT" \
# 			--save_dir $exp_path \
# 			--seed $i \
# 			--pooling "avg" \
# 			--output_merge "None" \
# 			--num_layers 6 \
# 			--attn_heads 10 \
# 			--num_epoch 65 \
# 			--shuffle \
# 			--name $i \
# 			--modify 1 \
# 			--aspect 0 \
# 			2>&1 | tee $exp_path/training.log

# 	done

CUDA_VISIBLE_DEVICES=0 python -u analysis.py \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--glove_dir $emb_dir \
	--model "RGAT" \
	--save_dir $exp_path \
	--seed 0 \
	--pooling "avg" \
	--output_merge "None" \
	--num_layers 6 \
	--attn_heads 10 \
	--num_epoch 65 \
	--shuffle \
	--modify 1 \
	--aspect 0 \
	--tfidf 0 \
	2>&1 | tee $exp_path/training.log

#42 39 31 15 
#7
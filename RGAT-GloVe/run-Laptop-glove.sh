#!/bin/bash
# training command for different datasets.
source_dir=../dataset
emb_dir=../dataset/Biaffine/glove
save_dir=saved_models

exp_setting=train

####### Laptops  acc:78.02 f1:74.00 #########
exp_dataset=Biaffine/glove/Laptops
exp_path=$save_dir/Laptops/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

END=50
for ((i=20;i<=30;i++)); 
	do 
		CUDA_VISIBLE_DEVICES=0 python -u train.py \
			--data_dir $source_dir/$exp_dataset \
			--vocab_dir $source_dir/$exp_dataset \
			--glove_dir $emb_dir \
			--model "RGAT" \
			--save_dir $exp_path \
			--seed $i \
			--pooling "avg" \
			--output_merge "gate" \
			--num_layers 4 \
			--attn_heads 5 \
			--num_epoch 60 \
			--shuffle \
			--modify 1 \
			--aspect 0 \
			2>&1 | tee $exp_path/training.log

		CUDA_VISIBLE_DEVICES=0 python -u train.py \
			--data_dir $source_dir/$exp_dataset \
			--vocab_dir $source_dir/$exp_dataset \
			--glove_dir $emb_dir \
			--model "RGAT" \
			--save_dir $exp_path \
			--seed $i \
			--pooling "avg" \
			--output_merge "gate" \
			--num_layers 4 \
			--attn_heads 5 \
			--num_epoch 60 \
			--shuffle \
			--modify 1 \
			--aspect 1 \
			2>&1 | tee $exp_path/training.log
	done

# CUDA_VISIBLE_DEVICES=0 python -u train.py \
# 	--data_dir $source_dir/$exp_dataset \
# 	--vocab_dir $source_dir/$exp_dataset \
# 	--glove_dir $emb_dir \
# 	--model "RGAT" \
# 	--save_dir $exp_path \
# 	--seed 18 \
# 	--pooling "avg" \
# 	--output_merge "gate" \
# 	--num_layers 4 \
# 	--attn_heads 5 \
# 	--num_epoch 60 \
# 	--shuffle \
# 	--modify 1 \
# 	--aspect 0 \
# 	--tfidf 0 \
# 	2>&1 | tee $exp_path/training.log

#18
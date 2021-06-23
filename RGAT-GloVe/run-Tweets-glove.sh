#!/bin/bash
# training command for different datasets.
source_dir=../dataset
emb_dir=../dataset/Biaffine/glove
save_dir=saved_models

exp_setting=train

####### Tweets  acc:75.36 f1:74.15 #########
exp_dataset=Biaffine/glove/Tweets
exp_path=$save_dir/Tweets/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

# END=50
# for ((i=8;i<=END;i++)); 
# 	do 
# 		CUDA_VISIBLE_DEVICES=0 python -u train.py \
# 			--data_dir $source_dir/$exp_dataset \
# 			--vocab_dir $source_dir/$exp_dataset \
# 			--glove_dir $emb_dir \
# 			--model "RGAT" \
# 			--save_dir $exp_path \
# 			--seed $i \
# 			--pooling "avg" \
# 			--output_merge "gate" \
# 			--num_layers 6 \
# 			--attn_heads 5 \
# 			--num_epoch 60 \
# 			--shuffle \
# 			--modify 0 \
# 			--aspect 0 \
# 			2>&1 | tee $exp_path/training.log

# 		CUDA_VISIBLE_DEVICES=0 python -u train.py \
# 			--data_dir $source_dir/$exp_dataset \
# 			--vocab_dir $source_dir/$exp_dataset \
# 			--glove_dir $emb_dir \
# 			--model "RGAT" \
# 			--save_dir $exp_path \
# 			--seed $i \
# 			--pooling "avg" \
# 			--output_merge "gate" \
# 			--num_layers 6 \
# 			--attn_heads 5 \
# 			--num_epoch 60 \
# 			--shuffle \
# 			--modify 1 \
# 			--aspect 0 \
# 			2>&1 | tee $exp_path/training.log

# 	done

CUDA_VISIBLE_DEVICES=0 python -u train.py \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--glove_dir $emb_dir \
	--model "RGAT" \
	--save_dir $exp_path \
	--seed 17 \
	--pooling "avg" \
	--output_merge "gate" \
	--num_layers 6 \
	--attn_heads 5 \
	--num_epoch 60 \
	--shuffle \
	--modify 1 \
	--aspect 0 \
	--tfidf 0 \
	2>&1 | tee $exp_path/training.log



#74.66

# 31 29 17
# Aspect-Specific Heterogeneous Relational Graph Attention Neural Networks for Aspect-Based Sentiment Analysis

This repository contains the code from the paper "[Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network](https://arxiv.org/abs/2002.09685)", IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)

## Setup

This code runs Python 3.6 with the following libraries:

+ Pytorch 1.2.0
+ Transformers 2.9.1
+ GTX 1080 Ti

You can also create an virtual environments with `conda` by run

```
conda env create -f requirements.yaml
```

## Get start

1. Prepare data

   + Restaurants, Laptop, Tweets and MAMS dataset. (We provide the parsed data at directory `dataset`)

   + Downloading Glove embeddings (available at [here](http://nlp.stanford.edu/data/glove.840B.300d.zip)), then  run 

     ```
     awk '{print $1}' glove.840B.300d.txt > glove_words.txt
     ```

     to get `glove_words.txt`.

2. Build vocabulary

   ```
   bash build_vocab.sh
   ```

3. Training
   Go to Corresponding directory and run scripts:

   ``` 
   bash run-MAMS-glove.sh
   bash run-MAMS-BERT.sh
   ```

4. The saved model and training logs will be stored at directory `saved_models`  

## References

```
@ARTICLE{bai21syntax,  
	author={Xuefeng Bai and Pengbo Liu and Yue Zhang},  
	journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
	title={Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network},   
	year={2021},  
	volume={29}, 
	pages={503-514},  
	doi={10.1109/TASLP.2020.3042009}
}
```




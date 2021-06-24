# encoding=utf-8
import sys
sys.path.append('../')
import torch
import random
import argparse
import numpy as np
from vocab import Vocab
from utils import helper
from sklearn import metrics
from loader import ABSADataLoader
from trainer import ABSATrainer
from load_w2v import load_pretrained_embedding
import wandb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import pickle
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt
import pydot
import tqdm
import matplotlib as mpl
import netgraph


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset/Restaurants")
parser.add_argument("--vocab_dir", type=str, default="dataset/Restaurants")
parser.add_argument("--glove_dir", type=str, default="dataset/glove")
parser.add_argument("--emb_dim", type=int, default=300, help="Word embedding dimension.")
parser.add_argument("--post_dim", type=int, default=30, help="Position embedding dimension.")
parser.add_argument("--pos_dim", type=int, default=30, help="Pos embedding dimension.")
parser.add_argument("--dep_dim", type=int, default=30, help="dep embedding dimension.")
parser.add_argument("--hidden_dim", type=int, default=50, help="hidden dim.")
parser.add_argument("--num_layers", type=int, default=2, help="Num of RGAT layers.")
parser.add_argument("--num_class", type=int, default=3, help="Num of sentiment class.")
parser.add_argument("--cross_val_fold", type=int, default=10, help="Num of cross valid class.")

parser.add_argument("--input_dropout", type=float, default=0.7, help="Input dropout rate.")
parser.add_argument("--layer_dropout", type=float, default=0, help="RGAT layer dropout rate.")
parser.add_argument(
    "--att_dropout", type=float, default=0, help="self-attention layer dropout rate."
)
parser.add_argument("--attn_heads", type=int, default=5, help="Num of GAT/RGAT attention heads.")
parser.add_argument("--alpha", type=float, default=1.0, help="Weight of structure attention.")
parser.add_argument("--beta", type=float, default=1.0, help="Weight of structure values.")
parser.add_argument("--lower", default=True, help="Lowercase all words.")
parser.add_argument("--direct", default=False)
parser.add_argument("--loop", default=True)


parser.add_argument("--bidirect", default=True, help="Do use bi-RNN layer.")
parser.add_argument("--rnn_hidden", type=int, default=50, help="RNN hidden state size.")
parser.add_argument("--rnn_layers", type=int, default=1, help="Number of RNN layers.")
parser.add_argument("--rnn_dropout", type=float, default=0.1, help="RNN dropout rate.")

parser.add_argument("--lr", type=float, default=0.01, help="learning rate.")
parser.add_argument(
    "--optim",
    choices=["sgd", "adagrad", "adam", "adamax"],
    default="adamax",
    help="Optimizer: sgd, adagrad, adam or adamax.",
)
parser.add_argument("--num_epoch", type=int, default=100, help="Number of total training epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
parser.add_argument("--log_step", type=int, default=20, help="Print log every k steps.")
parser.add_argument("--log", type=str, default="logs.txt", help="Write training log to file.")
parser.add_argument("--save_dir", type=str, default="./saved_models", help="Root dir for saving models.")
parser.add_argument("--model", type=str, default="std", help="model to use, (std, GAT, RGAT)")
parser.add_argument(
    "--pooling", type=str, default="avg", help="pooling method to use, (avg, max, attn)"
)
parser.add_argument(
    "--output_merge", type=str, default="gate", help="merge method to use, (addnorm, add, attn)"
)
parser.add_argument("--shuffle", default=False, action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tune", default=False, action="store_true")
parser.add_argument("--wandb", default=False)
parser.add_argument("--modify", type=int, default=0)
parser.add_argument("--aspect", type=int, default=0)
parser.add_argument("--tfidf", type=int, default=0)
parser.add_argument("--name", type=str)
args = parser.parse_args()


# load data
def get_dataloaders(args, vocab):
    train_batch = ABSADataLoader(
        args.data_dir + "/train_v1.json", args.batch_size, args, vocab, shuffle=args.shuffle
    )
    valid_batch = ABSADataLoader(
        args.data_dir + "/valid_v1.json", args.batch_size, args, vocab, shuffle=False
    )
    test_batch = ABSADataLoader(
        args.data_dir + "/test_v1.json", args.batch_size, args, vocab, shuffle=False
    )
    return train_batch, valid_batch, test_batch

def accuracy_score_by_class(predict, target):
    train_matrix = confusion_matrix(target, predict)
    train_accs = train_matrix.diagonal()/train_matrix.sum(axis=1)
    return train_accs


def evaluate(model, data_loader):
    predictions, labels = [], []
    val_loss, val_acc, val_step = 0.0, 0.0, 0
    val_neg_loss, val_neu_loss, val_pos_loss = 0.0, 0.0, 0
    for i, batch in enumerate(data_loader):
        loss, loss_by_class, acc, pred, label, _, _ = model.predict(batch)
        val_neg_loss += loss_by_class[0]
        val_neu_loss += loss_by_class[1]
        val_pos_loss += loss_by_class[2]
        val_loss += loss
        val_acc += acc
        predictions += pred
        labels += label
        val_step += 1
    # f1 score
    f1_score = metrics.f1_score(labels, predictions, average="macro")
    # accuracy
    test_acc_by_class = accuracy_score_by_class(predictions, labels)
    return val_loss / val_step, val_neg_loss / val_step, val_neu_loss / val_step, val_pos_loss / val_step, \
           val_acc / val_step, f1_score, test_acc_by_class,


def _totally_parameters(model):  #
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


# load vocab
print("Loading vocab...")
token_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_tok.vocab")  # token
post_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_post.vocab")  # position
pos_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pos.vocab")  # POS
dep_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_dep.vocab")  # deprel
pol_vocab = Vocab.load_vocab(args.vocab_dir + "/vocab_pol.vocab")  # polarity
vocab = (token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab)
print(
    "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(
        len(token_vocab), len(post_vocab), len(pos_vocab), len(dep_vocab), len(pol_vocab)
    )
)
args.tok_size = len(token_vocab)
args.post_size = len(post_vocab)
args.pos_size = len(pos_vocab)
args.dep_size = len(dep_vocab)

# print(dep_vocab.itos)

# load pretrained word emb
print("Loading pretrained word emb...")
word_emb = load_pretrained_embedding(glove_dir=args.glove_dir, word_list=token_vocab.itos)
assert len(word_emb) == len(token_vocab)
assert len(word_emb[0]) == args.emb_dim
word_emb = torch.FloatTensor(word_emb)  # convert to tensor

def test(path):    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_batch, valid_batch, test_batch = get_dataloaders(args, vocab)

    trainer = ABSATrainer(args, emb_matrix=word_emb)
    best_path = "/home/wuharlem/projects/paper/RGAT-ABSA/RGAT-GloVe/saved_models/Restaurants/train"
    trainer = torch.load(best_path + path)

    val_loss, val_neg_loss, val_neu_loss, val_pos_loss, val_acc, val_f1, val_acc_by_class = evaluate(trainer, test_batch)
    print("Evaluation Results: test_loss:{}, test_acc:{}, test_f1:{}".format(val_loss, val_acc, val_f1))

if __name__ == "__main__":

    # torch.nn.Module.dump_patches = True

    # for i in range(16, 31):
        # test(f'/modify/v4/best_checkpoint_modify_aspect_{i}.pt')
        # test(f'/modify/v4/best_checkpoint_modify_{i}.pt')
    test(f'/modify/best_checkpoint_modify_0.pt')
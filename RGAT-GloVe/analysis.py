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


# load pretrained word emb
print("Loading pretrained word emb...")
word_emb = load_pretrained_embedding(glove_dir=args.glove_dir, word_list=token_vocab.itos)
assert len(word_emb) == len(token_vocab)
assert len(word_emb[0]) == args.emb_dim
word_emb = torch.FloatTensor(word_emb)  # convert to tensor

def ablation(batch_num, select_num, att_num, path, folder, all=None):
    ## set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
        
    ## load dataset
    _, valid_batch, _ = get_dataloaders(args, vocab)

    ## load model
    trainer   = ABSATrainer(args, emb_matrix=word_emb)
    best_path = "/home/wuharlem/projects/paper/RGAT-ABSA/RGAT-GloVe/saved_models/Restaurants/train"
    trainer   = torch.load(best_path + path)
    wrong_index = []       
    att_record  = {}
    edge_num_record = {}

    ## pick which batch
    for num, batch in enumerate(valid_batch):
        if batch_num or select_num:
            if num!=batch_num:
                continue

        ## into cuda
        batch = [b.cuda() for b in batch]

        # unpack inputs and label
        inputs = batch[0:-1]
        label  = batch[-1]

        # forward
        trainer.model.eval()
        logits, _, all_attention = trainer.model(inputs)
        tok, asp, pos, head, deprel, post, mask_ori, win, tfidf, tfidf_flag, seq_len = inputs  # unpack inputs

        # choose which attention
        top_attn = all_attention[att_num]

        for batch_idx in range(len(label)):

            ## save data
            att_record[(num, batch_idx)] = {dep_id:{0:0, 1:0} for dep_id in range(len(dep_vocab.itos))}
            att_record[(num, batch_idx)]['[s]'] = {0:0, 1:0}
            att_record[(num, batch_idx)]['[m]'] = {0:0, 1:0}
            att_record[(num, batch_idx)]['[w]'] = {0:0, 1:0}
            edge_num_record[(num, batch_idx)] = {dep_id:0 for dep_id in range(len(dep_vocab.itos))}
            edge_num_record[(num, batch_idx)]['[s]'] = 0
            edge_num_record[(num, batch_idx)]['[m]'] = 0
            edge_num_record[(num, batch_idx)]['[w]'] = 0

            if select_num:
                if batch_idx!=select_num:
                    continue
            else:
                if int(label[batch_idx])!=int(torch.argmax(logits[batch_idx])):
                    wrong_index.append((num, batch_idx))
                else:
                    pass

            ## print Data Info
            LEN = len(tok[batch_idx])
            logits_str = [float(f"{logit:.2f}") for logit in logits[batch_idx].tolist()]
            graph_log = f"ASPECT: {' '.join([token_vocab.itos[int(asp[batch_idx][i])] for i in range(len(asp[batch_idx])) if int(asp[batch_idx][i])])}\nLABEL: {pol_vocab.itos[int(label[batch_idx])], int(label[batch_idx])}\nOUTPUT: {logits_str}\nTEXT: {' '.join([token_vocab.itos[int(tok[batch_idx][i])] for i in range(LEN) if int(tok[batch_idx][i])])}"
            print(graph_log)

            ## Initialize
            node_labels          = {}
            edge_labels          = {}
            mutual_edge_labels   = {}
            win_edge_labels      = {}
            selfloop_edge_labels = {}
            aspect_edge_labels   = {}

            edges = []
            selfloop_edges = []

            edge_alphas = {}
            
            pad_num = 0

            ## Build Nodes
            for i in range(LEN):
                if int(tok[batch_idx][i]):
                    if args.modify:
                        node_labels[i+1] = token_vocab.itos[int(tok[batch_idx][i])]
                    else:
                        node_labels[i]   = token_vocab.itos[int(tok[batch_idx][i])] 
                else:
                    ## remove <PAD>
                    pad_num+=1
            if args.modify:
                node_labels[0] = 's'
                node_labels[LEN+1] = 'a'

            for i in range(LEN):
                deprel_ids = deprel[batch_idx][i]
                input_ids  = tok[batch_idx][i]

                ## get text string
                token     = " "*8
                if len(token_vocab.itos[int(input_ids)]) < 8:
                    token = token_vocab.itos[int(input_ids)][:len(token_vocab.itos[int(input_ids)])] + token[:8-len(token_vocab.itos[int(input_ids)])]
                else:
                    token = token_vocab.itos[int(input_ids)]

                ## get (v, u) by ids
                if args.modify:
                    v = head[batch_idx][i].item()
                    u = i+1
                else:
                    v = head[batch_idx][i].item()-1
                    u = i

                ## logs
                # print(i, '\t', token, '\t', 
                #       int(mask_ori[batch_idx][i].item()), '\t',  
                #       int(tfidf[batch_idx][i]),'\t',
                #       ((v, u), dep_vocab.itos[int(deprel_ids)])
                # ) 

                ## SELFLOOP
                if args.loop and int(deprel_ids):

                    # Add edges
                    selfloop_edges.append((i, i))

                    # Add labels
                    selfloop_edge_labels[(i, i)] = f"[s] \n ({top_attn[batch_idx][i][i]:.1f})"

                    # record
                    edge_num_record[(num, batch_idx)]["[s]"] +=2
                    att_record[(num, batch_idx)]["[s]"][0] += float(top_attn[batch_idx][i][i])
                    att_record[(num, batch_idx)]["[s]"][1] += float(top_attn[batch_idx][i][i])

                    # if top_attn[batch_idx][i][i]>0.01:
                    #     selfloop_edge_labels[(i, i)] = f"[s] \n ({top_attn[batch_idx][i][i]:.3f})"
                    # else:
                    #     selfloop_edge_labels[(i, i)] = f"[s]"
                
                if head[batch_idx][i].item() == 0:  ## root
                    continue
                elif int(deprel_ids): ## has edge
                    
                    # Add edges
                    edges.append((v, u))
                    # edges.append((u, v))
                    
                    # Add labels
                    edge_labels[(v, u)] = f"{dep_vocab.itos[int(deprel_ids)]} \n ({top_attn[batch_idx][v][u]:.1f}|{top_attn[batch_idx][u][v]:.1f})"

                    # record
                    edge_num_record[(num, batch_idx)][int(deprel_ids)] +=2
                    att_record[(num, batch_idx)][int(deprel_ids)][0] += float(top_attn[batch_idx][v][u])
                    att_record[(num, batch_idx)][int(deprel_ids)][1] += float(top_attn[batch_idx][u][v])
                    edge_alphas[(v, u)] = float(top_attn[batch_idx][v][u])
                    edge_alphas[(u, v)] = float(top_attn[batch_idx][u][v])

                    # if top_attn[batch_idx][v][u]>=0.01 or top_attn[batch_idx][u][v]>=0.01:
                        # edge_labels[(v, u)] = f"{dep_vocab.itos[int(deprel_ids)]} \n ({top_attn[batch_idx][v][u]:.2f})"
                        # edge_labels[(u, v)] = f"{dep_vocab.itos[int(deprel_ids)]} \n ({top_attn[batch_idx][u][v]:.2f})"
                    # else:
                        # edge_labels[(v, u)] = f"{dep_vocab.itos[int(deprel_ids)]}"
                        # edge_labels[(u, v)] = f"{dep_vocab.itos[int(deprel_ids)]}"

            ## Edge between aspect
            for i in torch.nonzero(mask_ori[batch_idx]):
                i = int(i)
                if args.modify:
                    i+=1
                for j in torch.nonzero(mask_ori[batch_idx]):
                    j = int(j)
                    if args.modify:
                        j+=1
                    if (i, j) in edges or (j, i) in edges or i==j:
                        pass
                    else:
                        ## Add edges
                        edges.append((i, j))
                        # edges.append((j, i))

                        ## Add labels
                        aspect_edge_labels[(i, j)] = f"[s] \n ({top_attn[batch_idx][i][j]:.1f}|{top_attn[batch_idx][j][i]:.1f})"

                        ## record
                        edge_alphas[(i, j)] = float(top_attn[batch_idx][i][j])
                        edge_alphas[(j, i)] = float(top_attn[batch_idx][j][i])
                        edge_num_record[(num, batch_idx)]['[s]'] +=2
                        att_record[(num, batch_idx)]['[s]'][0] += float(top_attn[batch_idx][i][j])
                        att_record[(num, batch_idx)]['[s]'][1] += float(top_attn[batch_idx][j][i])

                        # if top_attn[batch_idx][j][i]>=0.01:
                        #     aspect_edge_labels[(i, j)] = f"[s] \n ({top_attn[batch_idx][i][j]:.3f})"
                        #     aspect_edge_labels[(j, i)] = f"[s] \n ({top_attn[batch_idx][j][i]:.3f})"
                        # else:
                        #     aspect_edge_labels[(i, j)] = f"[s]"
                        #     aspect_edge_labels[(j, i)] = f"[s]"

            if args.modify:
                ## special token
                for i in torch.nonzero(win[batch_idx]):
                    i = int(i)

                    ## Add edges
                    edges.append((LEN+1, i+1))
                    # edges.append((i+1, LEN+1))

                    ## Add labels
                    win_edge_labels[(LEN+1, i+1)] = f"[w] \n ({top_attn[batch_idx][-1][i+1]:.1f}|{top_attn[batch_idx][i+1][-1]:.1f})"

                    ## record
                    edge_num_record[(num, batch_idx)]['[w]'] +=2
                    att_record[(num, batch_idx)]["[w]"][0] += float(top_attn[batch_idx][-1][i+1])
                    att_record[(num, batch_idx)]["[w]"][1] += float(top_attn[batch_idx][i+1][-1])
                    edge_alphas[(LEN+1, i+1)] = float(top_attn[batch_idx][-1][i+1])
                    edge_alphas[(i+1, LEN+1)] = float(top_attn[batch_idx][i+1][-1])

                    # if top_attn[batch_idx][-1][i+1]>=0.01 or top_attn[batch_idx][i+1][-1]>=0.01:
                    #     win_edge_labels[(LEN+1, i+1)] = f"[w] \n ({top_attn[batch_idx][-1][i+1]:.2f})"
                    #     win_edge_labels[(i+1, LEN+1)] = f"[w] \n ({top_attn[batch_idx][i+1][-1]:.2f})"
                    # else:
                    #     win_edge_labels[(LEN+1, i+1)] = f"[w]"
                    #     win_edge_labels[(i+1, LEN+1)] = f"[w]"

                ## Add edges
                edges.append((0, LEN+1)) ## Mutual Indication
                # edges.append((LEN+1, 0)) ## Mutual Indication

                ## Add labels
                mutual_edge_labels[(0, LEN+1)] = f"[m] \n ({top_attn[batch_idx][0][-1]:.1f}|{top_attn[batch_idx][-1][0]:.1f})"

                ## record
                edge_num_record[(num, batch_idx)]['[m]'] +=2
                att_record[(num, batch_idx)]["[m]"][0] += float(top_attn[batch_idx][0][-1])
                att_record[(num, batch_idx)]["[m]"][1] += float(top_attn[batch_idx][-1][0])
                edge_alphas[(0, LEN+1)] = float(top_attn[batch_idx][0][-1])
                edge_alphas[(LEN+1, 0)] = float(top_attn[batch_idx][-1][0])

                # mutual_edge_labels[(0, LEN+1)] = f"[m] \n ({top_attn[batch_idx][0][-1]:.2f})"
                # mutual_edge_labels[(LEN+1, 0)] = f"[m] \n ({top_attn[batch_idx][-1][0]:.2f})"

                # if args.tfidf:
                #     ## TFIDF
                #     for ii in range(LEN):
                #         if not int(tok[batch_idx][ii]):
                #             continue
                #         edges.append((0, ii))
                #         if top_attn[batch_idx][0][ii] > 0.01 or top_attn[batch_idx][ii][0] > 0.01:
                #             special_edge_labels[(0, ii)] = f"{int(tfidf[batch_idx][ii])} ({top_attn[batch_idx][0][ii]:.3f}, {top_attn[batch_idx][ii][0]:.3f})"

            att_record[(num, batch_idx)] = {
                key: [(att_record[(num, batch_idx)][key][0]+1e-10)/(edge_num_record[(num, batch_idx)][key]/2+1e-10), 
                      (att_record[(num, batch_idx)][key][1]+1e-10)/(edge_num_record[(num, batch_idx)][key]/2+1e-10)] \
                for key in att_record[(num, batch_idx)]}

            # draw
            f  = plt.figure(figsize=(10, 10))
            ax = f.add_axes([0, 0, 1, 1])
            plt.text(0.01, 0.01, graph_log.replace('$', ''), ha='left', va='bottom', transform=ax.transAxes)
            G = nx.DiGraph()

            G.add_nodes_from(range(0, LEN))
            G.remove_nodes_from(range(LEN-pad_num, LEN))  ## remove pad
            G.add_edges_from(edges)

            pos   = graphviz_layout(G, prog="dot")

            nx.draw(G, pos, labels=node_labels, 
                    edge_color = 'white', width = 0.3, 
                    with_labels = True, 
                    node_size=1000, 
                    node_color = 'white', 
                    font_color = 'black', 
                    alpha=0.9)

            draw_edges = nx.draw_networkx_edges(G, pos, node_size=1000)

            draw_edges_label = nx.draw_networkx_edge_labels(G, pos, bbox=dict(alpha=0), edge_labels=edge_labels, font_color='blue')
            draw_edges_label_a = nx.draw_networkx_edge_labels(G, pos, bbox=dict(alpha=0), edge_labels=aspect_edge_labels, font_color='green')

            if args.modify:
                draw_edges_label_m = nx.draw_networkx_edge_labels(G, pos, bbox=dict(alpha=0), edge_labels=mutual_edge_labels, font_color='orange')
                draw_edges_label_w = nx.draw_networkx_edge_labels(G, pos, bbox=dict(alpha=0), edge_labels=win_edge_labels, font_color='red')

            # set alpha value for each edge
            # for i, ee in enumerate(G.edges):
            #     draw_edges[i].set_alpha(edge_alphas[ee])
            #     if ee in draw_edges_label:
            #         draw_edges_label[ee].set_alpha(edge_alphas[ee])
            #     if ee in draw_edges_label_a:
            #         draw_edges_label_a[ee].set_alpha(edge_alphas[ee])
            #     if args.modify and ee in draw_edges_label_m:
            #         draw_edges_label_m[ee].set_alpha(edge_alphas[ee])
            #     if args.modify and ee in draw_edges_label_w:
            #         draw_edges_label_w[ee].set_alpha(edge_alphas[ee])

            # Save
            if args.modify:
                plt.savefig(f"graph/Restaurant/{folder}/Graph_{num}_{batch_idx}_{att_num}_modify.png", format="PNG")
            else:
                plt.savefig(f"graph/Restaurant/{folder}/Graph_{num}_{batch_idx}_{att_num}.png", format="PNG")

    # with open("wrong_index_modify.txt", "w") as file:
    #     for i in wrong_index:
    #         b, n = i
    #         file.write(f"{b}, {n}\n")

    # head = 9
    # with open(f'Analysis/pkl/{head}/att_{att_num}.pkl', 'wb') as handle:
    #     pickle.dump(att_record, handle)
        
    # with open(f'Analysis/pkl/{head}/num_{att_num}.pkl', 'wb') as handle:
    #     pickle.dump(edge_num_record, handle)

if __name__ == "__main__":

    # torch.nn.Module.dump_patches = True

    # ## Ablation Study
    # with open("wrong_index.txt", "r") as file:
    #     wrong_index = file.read().replace(" ", "").split()

    # with open("wrong_index_modify.txt", "r") as file:
    #     wrong_index_modify = file.read().replace(" ", "").split()

    # num = 0
    # max_num = num+10
    # i = 0
    # for index in wrong_index:
    #     if index not in wrong_index_modify:
    #         if i >= num:
    #             if num > max_num: 
    #                 break
    #             a, b = index.split(',')
    #             # print(a,b)
    #             # ablation(int(a), int(b), 5, f'/original/best_checkpoint_v2_7.pt', "compare")
    #             # ablation(int(a), int(b), 5, f'/modify/v3/best_checkpoint_modify_7.pt', "compare")
    #             num+=1
    #         i+=1
    
    
    ablation(3, 27, 5, f'/modify/best_checkpoint_modify_0.pt', "3-27")
    # 0 28, 0 20, 1 13, 1 14, 1 15, 1 19, 1 21, 1 24

    # for att in range(6):
    #     ablation(14, 9, att, f'/modify/v3/best_checkpoint_modify_7.pt', "14-9")
        # ablation(15, 19, att, f'/original/best_checkpoint_v2_7.pt')
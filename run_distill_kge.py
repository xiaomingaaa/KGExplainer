import os
import argparse
from distillation.dataloader_distill import TrainDataset
from distillation.kge_distillation import HeteroGAT
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, SGD
from tqdm import tqdm
import pickle

torch.manual_seed(6789)
np.random.seed(6789)
torch.cuda.manual_seed_all(6789)
os.environ['PYTHONHASHSEED'] = str(6789)

def norm(x):
    x = F.normalize(x, p=-1)
    return x

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Negative sample for Distillation',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--dataset', type=str,default='family', help='used dataset')
    parser.add_argument('--model_name', type=str,default='TransE', help='used dataset')
    parser.add_argument('--n_size', type=int,default=3, help='negative sample size')
    parser.add_argument('--batch_size', type=int,default=1024, help='batch size')
    parser.add_argument('--k_hop', type=int,default=2, help='hops of neighbors')
    parser.add_argument('--num_steps', type=int,default=10000, help='steps to train')
    args = parser.parse_args(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return args

def main(args):
    dataset = TrainDataset(args)
    
    model = HeteroGAT(dataset.graph.etypes, dataset.graph.num_nodes('node'), 32, 64, 32)
    model.to(args.device)
    optim = Adam(model.parameters(), lr=0.001)
    loss_func = F.mse_loss
    margin_loss = nn.MarginRankingLoss(12, reduction='sum')
    # loss_func = F.kl_div
    print(dataset.graph.etypes, dataset.graph.num_nodes('node'))
    pos_g, neg_g, pos_embeds, neg_embeds, pos_scores, neg_scores = dataset.read_samples()
    bar = tqdm(range(args.num_steps))
    for step in bar:
        # g, real_scores = dataset.rand_generate_samples()
        # bar = tqdm(range(int(args.batch_size/2)))
        embed_pos, score_pos = model(pos_g)
        embed_neg, score_neg = model(neg_g)
        # print(score, embed)
        # import ipdb;ipdb.set_trace()
        loss_embed_pos = loss_func(pos_embeds, embed_pos)
        loss_embed_neg = loss_func(neg_embeds, embed_neg)
        loss = margin_loss(score_pos, score_neg, torch.ones(len(score_pos), 1).to(args.device)) + loss_embed_pos + loss_embed_neg
        optim.zero_grad()
        loss.backward()
        optim.step()
        bar.set_description(f'Training: step-{step+1} | loss: {loss.detach().cpu().numpy()}')
        # print(f'step {step+1}: loss: {loss.detach().cpu().numpy()}')

        # if (step+1) % 500==0:
    
    torch.save(model, f'datasets/{args.dataset}/{args.dataset}_model.pt')

def preprocess(args):
    dataset = TrainDataset(args)
    dataset.save_samples()

if __name__=='__main__':
    args = parse_args()
    main(args)
    # preprocess(args)


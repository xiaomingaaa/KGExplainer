#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from .construct_samples import read_triple
import random
import dgl
from collections import defaultdict
import time
from tqdm import tqdm
import pickle
import torch.nn.functional as F
import multiprocessing as mp

from torch.utils.data import Dataset

class TrainDataset():
    def __init__(self, args):
        self.args = args
        self.graph, self.entity2id, self.relation2id = self._construct_dgl_graph()
        # self.samples = self._read_samples()
        self.entity_embed = torch.from_numpy(np.load(f'ckpts/{args.model_name}_{args.dataset}_1/entity_embedding.npy'))
        self.relation_embed = torch.from_numpy(np.load(f'ckpts/{args.model_name}_{args.dataset}_1/relation_embedding.npy'))
        

    def _construct_dgl_graph(self):
        with open(f'datasets/{self.args.dataset}/entities.dict') as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(f'datasets/{self.args.dataset}/relations.dict') as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        train_triples = read_triple(f'datasets/{self.args.dataset}/train.txt', entity2id, relation2id)
        edge_dict = defaultdict(list)
        for (h, r, t) in train_triples:
            etype = ('node', str(r), 'node')
            edge_dict[etype] += [(h, t)]
        # graph = dgl.graph(num_nodes=len(entity2id))
        graph = dgl.heterograph(edge_dict, num_nodes_dict={'node':len(entity2id)})
        # graph = dgl.add_reverse_edges(graph).to(self.args.device)
        # graph = graph.to(self.args.device)
        
        return  graph, entity2id, relation2id
        
    def _read_samples(self):
        file_path = f'datasets/{self.args.dataset}/distill_samples.txt'
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                infos = line.strip().split('\t')
                h, r, t, score = int(infos[0]), int(infos[1]), int(infos[2]), float(infos[3])
                samples.append([h,r,t,score])
        
        return samples

    def _read_samples_pos(self):
        file_path = f'datasets/{self.args.dataset}/test.txt'
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                infos = line.strip().split('\t')
                h, r, t= int(infos[0]), int(infos[1]), int(infos[2])
                samples.append([h,r,t])
        
        return samples

    def _read_condidate_entity(self):
        file_path = f'datasets/{self.args.dataset}/test.txt'
        condidate_entity = []
        with open(file_path, 'r') as f:
            for line in f:
                infos = line.strip().split('\t')
                h, r, t= int(self.entity2id[infos[0]]), int(self.relation2id[infos[1]]), int(self.entity2id[infos[2]])
                condidate_entity.append(h)
                condidate_entity.append(t)
        
        return condidate_entity
        
    def _dist_embed(self, h, r, t, model_name='TransE'):
        if model_name=='TransE':
            h, t = self.entity_embed[h], self.entity_embed[t]
            r = self.relation_embed[r]
            embed = h + r - t
            score = torch.Tensor(12) - torch.norm(embed, p=1, dim=-1)
            return embed, score
        elif model_name=='DistMult':
            h, t = self.entity_embed[h], self.entity_embed[t]
            r = self.relation_embed[r]
            embed =  h * r * t
            score = embed.sum(dim = -1)
            return embed, score
        elif model_name=='RotatE':
            h, t = self.entity_embed[h], self.entity_embed[t]
            r = self.relation_embed[r]
            embed =  h * r * t
            pi = 3.14159265358979323846
    
            re_head, im_head = torch.chunk(h, 2, dim=2)
            re_tail, im_tail = torch.chunk(t, 2, dim=2)
            #Make phases of relations uniformly distributed in [-pi, pi]
            phase_relation = r/(self.embedding_range.item()/pi)
            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)
            
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

            embed = torch.stack([re_score, im_score], dim = 0)
            embed = score.norm(dim = 0)
            score = self.gamma.item() - score.sum(dim = 2)
            return embed, score

    def _score(self, h, r, t):
        h, t = self.entity_embed[h], self.entity_embed[t]
        r = self.relation_embed[r]
        score = h + r - t
        score = torch.Tensor(12) - torch.norm(score, p=1, dim=0)
        score = F.logsigmoid(score)
        return 

    def _get_subgraph(self, h, r, t):
        h_nodes, t_nodes = [], []
        for i in range(self.args.k_hop):
            h_in_subgraph = dgl.khop_in_subgraph(self.graph, {'node':torch.tensor([h])}, relabel_nodes=False, k=i+1)
            t_in_subgraph = dgl.khop_in_subgraph(self.graph, {'node':torch.tensor([t])}, relabel_nodes=False, k=i+1)
            for etype in h_in_subgraph.etypes:
                e = h_in_subgraph.edges(etype=etype)
                h_nodes += e[0].tolist()
                h_nodes += e[1].tolist()
            
            for etype in t_in_subgraph.etypes:
                e = t_in_subgraph.edges(etype=etype)
                t_nodes += e[0].tolist()
                t_nodes += e[1].tolist()

        # h_nodes, t_nodes = h_nodes.tolist(), t_nodes.tolist()
        inter_nodes = set(h_nodes).intersection(set(t_nodes))
        inter_nodes = inter_nodes.union(set([h,t]))
        h_sg = dgl.node_subgraph(self.graph, {'node':list(inter_nodes)})
        
        return h_sg

    def _generate_subgraph(self, batch_samples):
        graphs = []
        scores = []
        # print(batch_samples)
        for h, r, t, s in batch_samples:
            h_sg = self._get_subgraph(h, r, t)
            graphs.append(h_sg)
            embed = self._dist_embed(h, r, t)
            scores.append(embed.unsqueeze(0))
        
        graphs = dgl.batch(graphs).to(self.args.device)
        # import ipdb;ipdb.set_trace()
        scores = torch.concat(scores, dim=0)
        
        return graphs, scores.float().to(self.args.device)
        # return 0, 0
    
    def _generate_subgraph_mp(self, batch_samples):
        # print(batch_samples)
        h, r, t, s = batch_samples
        h_sg = self._get_subgraph(h, r, t)
        embed = self._dist_embed(h, r, t)
        # scores.append(embed.unsqueeze(0))
        
        # graphs = dgl.batch(graphs)
        # import ipdb;ipdb.set_trace()
        # scores = torch.concat(scores, dim=0)
        
        return h_sg, embed.unsqueeze(0)

    def _generate_subgraph_with_shortest_paths(self, batch_samples):
        graphs = []
        for [h, r, t, s] in batch_samples:
            pass

    def _generate_subgraph_with_negative(self, batch_samples):
        samples = []
        condidate_entity = self._read_condidate_entity()
        bar = tqdm(enumerate(batch_samples))
        for idx, [h, r, t] in bar:
            hg_pos = self._get_subgraph(h, r, t)
            t_ = random.choice(condidate_entity)
            while t_== h or t_ == t:
                t_ = random.choice(condidate_entity)
            hg_neg = self._get_subgraph(h, r, t_)
            # graphs.append(h_gs)
            samples.append([hg_pos, hg_neg, h, r, t, t_])
            bar.set_description(f'{idx+1}/{len(batch_samples)}')
        pickle.dump(samples, open(f'datasets/{self.args.dataset}/{self.args.dataset}_{self.args.k_hop}_samples.pkl', 'wb'))

    def rand_generate_samples(self):
        batch_size = self.args.batch_size
        start_time = time.time()
        batch_samples = random.choices(self.samples,k=batch_size)
        time_gap = (time.time()-start_time) / 1000
        print('sample time costing: ', time_gap) 
        start_time = time.time()
        graphs, real_scores = self._generate_subgraph(batch_samples)
        time_gap = (time.time()-start_time) / 1000
        print('subgraph sampling time costing: ', time_gap)

        return graphs, real_scores

    def generate_subgraph(self, batch_edges):
        graphs, real_scores = self._generate_subgraph(batch_edges)
        
        return graphs, real_scores

    def generate_subgraph_mp(self, batch_edges):
        graphs = []
        embeds = []
        with mp.Pool(processes=None) as p:
            for g, embed in tqdm(p.imap(self._generate_subgraph_mp, batch_edges), total=len(batch_edges)):
                graphs.append(g)
                embeds.append(embed)
        
        return graphs, embeds

    def save_samples(self):
        samples = []
        with open(f'datasets/{self.args.dataset}/train.txt', 'r') as f:
            for idx, line in enumerate(f):
                # if idx==1:
                #     break
                h, r, t = line.strip().split('\t') 
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                samples.append([h, r, t])
        
        self._generate_subgraph_with_negative(samples)
    
    def read_samples(self):
        samples = pickle.load(open(f'datasets/{self.args.dataset}/{self.args.dataset}_{self.args.k_hop}_samples.pkl', 'rb'))
        pos_graphs = []
        neg_graphs = []
        pos_embeds = []
        neg_embeds = []
        pos_scores = []
        neg_scores = []
        for [pos, neg, h, r, t, t_] in samples:
            pos_graphs.append(pos)
            neg_graphs.append(neg)
            pos_embed, pos_score = self._dist_embed(h, r, t, model_name=self.args.model_name)
            pos_embeds.append(pos_embed.unsqueeze(0))
        
            pos_score = F.logsigmoid(pos_score)
            pos_scores.append(pos_score)

            neg_embed, neg_score = self._dist_embed(h, r, t, model_name=self.args.model_name)
            neg_embeds.append(neg_embed.unsqueeze(0))
            
            neg_score = F.logsigmoid(neg_score)
            neg_scores.append(neg_score)
        
        pos_scores = torch.concat(pos_scores, dim=0).to(self.args.device)
        neg_scores = torch.concat(neg_scores, dim=0).to(self.args.device)
        pos_embeds = torch.concat(pos_embeds, dim=0).to(self.args.device)
        neg_embeds = torch.concat(neg_embeds, dim=0).to(self.args.device)
        pos_graphs = dgl.batch(pos_graphs).to(self.args.device)
        neg_graphs = dgl.batch(neg_graphs).to(self.args.device)

        return  pos_graphs, neg_graphs, pos_embeds, neg_embeds, pos_scores, neg_scores




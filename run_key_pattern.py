#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from key_subgraph.kge_model import KGEModel

from key_subgraph.dataloader import TrainDataset
from key_subgraph.dataloader import BidirectionalOneShotIterator
from collections import defaultdict
import dgl
import torch.nn.functional as F
from tqdm import tqdm

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', type=bool, default=True, help='use GPU')
    
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='family')
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=4096, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=1, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=10000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        # import ipdb;ipdb.set_trace()
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

def construct_dgl_graph(args):
    with open(f'datasets/{args.dataset}/entities.dict') as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    with open(f'datasets/{args.dataset}/relations.dict') as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    train_triples = read_triple(f'datasets/{args.dataset}/train.txt', entity2id, relation2id)
    
    edge_dict = defaultdict(list)
    for (h, r, t) in train_triples:
        etype = ('node', str(r), 'node')
        edge_dict[etype] += [(h, t)]
    
    graph = dgl.heterograph(edge_dict, num_nodes_dict={'node':len(entity2id)})
    
    return  graph, entity2id, relation2id

def get_subgraph(args, graph, h, r, t):
    outer_nodes = []
    h_khop_nodes = defaultdict(set)
    t_khop_nodes = defaultdict(set)
    for i in range(args.k_hop):
        h_nodes, t_nodes = [], []
        h_in_subgraph = dgl.khop_in_subgraph(graph, {'node':torch.tensor([h])}, relabel_nodes=False, k=i+1)
        t_in_subgraph = dgl.khop_in_subgraph(graph, {'node':torch.tensor([t])}, relabel_nodes=False, k=i+1)
        for etype in h_in_subgraph.etypes:
            e = h_in_subgraph.edges(etype=etype)
            h_nodes += e[0].tolist()
            h_nodes += e[1].tolist()
        
        for etype in t_in_subgraph.etypes:
            e = t_in_subgraph.edges(etype=etype)
            t_nodes += e[0].tolist()
            t_nodes += e[1].tolist()
        
        h_khop_nodes[i]=set(h_nodes)
        t_khop_nodes[i]=set(t_nodes)
    # h_nodes, t_nodes = h_nodes.tolist(), t_nodes.tolist()
    inter_nodes = set(h_khop_nodes[0]).union(set(t_khop_nodes[0]))
    inter_nodes = inter_nodes.union(set(h_khop_nodes[1]).intersection(set(t_khop_nodes[1])))
    inter_nodes = inter_nodes.union(set([h,t]))

    outer_subgraph = dgl.in_subgraph(graph, {'node':list(inter_nodes)}, relabel_nodes=False)
    for etype in outer_subgraph.etypes:
        e = outer_subgraph.edges(etype=etype)
        outer_nodes += e[0].tolist()
        outer_nodes += e[1].tolist()
    all_nodes = set(outer_nodes).union(inter_nodes)
    outer_nodes = set(outer_nodes).difference(inter_nodes)
    
    h_sg = dgl.node_subgraph(graph, {'node':list(all_nodes)}, relabel_nodes=False)
    
    return h_sg, outer_nodes, inter_nodes

def read_subgraph_kg(args, g, triple):
    h, r, t = triple
    subgraph, outer_nodes, inter_nodes = get_subgraph(args, g, h, r, t)
    subgraph_entity2id, subgraph_relation2id, subgraph_triples = {}, {}, []
    node_neighbors = defaultdict(set)
    for etype in subgraph.etypes:
        e = subgraph.edges(etype=etype)
        h = e[0].tolist()
        t = e[1].tolist()
        r = int(etype)
        for i in range(len(h)):
            node_neighbors[h[i]].add(t[i])
            node_neighbors[t[i]].add(h[i])
        # if h not in subgraph_entity2id:
        #     subgraph_entity2id[h] = len(subgraph_entity2id)
        # if t not in subgraph_entity2id:
        #     subgraph_entity2id[t] = len(subgraph_entity2id)
        # if r not in subgraph_relation2id:
        #     subgraph_relation2id[r] = len(subgraph_relation2id)

        # subgraph_triples.append([subgraph_entity2id[h], subgraph_relation2id[r], subgraph_entity2id[t]])
            subgraph_triples.append([h[i], r, t[i]])
    
    return subgraph_triples, list(outer_nodes), list(inter_nodes), node_neighbors

def drop_triples(triples, node):
    ts = []
    for [h, r, t] in triples:
        if h == node or t == node:
            continue
        ts.append((h, r, t))
    
    return ts

def importance_eval(subgraph_triples, nentity, nrelation, args, kge_model, optimizer, outer_nodes, fixed_embed, h, r, t):
    train_dataloader_head = DataLoader(
        TrainDataset(subgraph_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )
    train_dataloader_tail = DataLoader(
        TrainDataset(subgraph_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    # scores = []
    for epoch in range(args.epoch):
        for step in range(args.max_steps):
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            # print(log)
            embed = kge_model.entity_embedding.clone()
            embed[outer_nodes] = fixed_embed
            kge_model.entity_embedding = torch.nn.Parameter(embed)
        score = kge_model.predict(h, r, t)
        # scores.append(score)
    # print(score)
    return score
        # print(scores)
        # if epoch > 5:
        #     if abs(scores[-5]-scores[-1]) <= 0.05:
        #         best_score = min(scores[-5:])
        #         return best_score, score

def eval_prediction(model, graphs):
    with torch.no_grad():
        embed, score_pre = model(graphs)
        score_pre.detach().cpu().view(1, -1)
        score =  F.logsigmoid(- torch.norm(embed, p=1, dim=-1)).view(1, -1)
        return score_pre, score

def run(args):
    graph, entity2id, relation2id = construct_dgl_graph(args)

    nentity = len(entity2id)
    nrelation = len(relation2id)
    args.nentity = nentity
    args.nrelation = nrelation
    model = torch.load(f'/root/{args.dataset}_model.pt')
    model.to(args.device)
    model.eval()

    pred_scores = []
    kge_scores = []
    with open(f'datasets/{args.dataset}/test.txt', 'r') as f:
        bar = tqdm(enumerate(f), total=2835)
        for idx, line in bar:
            # if idx != 294:
            #     continue
            h, r, t = line.strip().split('\t')
            h, r, t = entity2id[h], relation2id[r], entity2id[t]
            subgraph_triples, outer_nodes, inter_nodes, node_neighs = read_subgraph_kg(args, graph, [h, r, t])
            # if len(node_neighs[h])==0 or len(node_neighs[t])==0:
            #     conti
            kge_model = KGEModel(
                model_name=args.model,
                nentity=nentity,
                nrelation=nrelation,
                hidden_dim=args.hidden_dim,
                gamma=args.gamma,
                double_entity_embedding=args.double_entity_embedding,
                double_relation_embedding=args.double_relation_embedding,
                dataset=args.dataset
            )

            if args.cuda:
                kge_model = kge_model.cuda()

            current_learning_rate = args.learning_rate
            optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
            )

            fixed_embed = kge_model.entity_embedding[outer_nodes]
            pre_score = kge_model.predict(h, r, t)
            finding_nodes = [h]
            visited_nodes = []
            key_nodes = [h]
            while len(finding_nodes) != 0:
                # print('finding nodes: ', finding_nodes)
                f_node = finding_nodes.pop()

                if t in node_neighs[f_node]:
                    # print('arrived!')
                    key_nodes.append(t)
                    break
                # print('current node: ', f_node)
                visited_nodes.append(f_node)
                new_score_list = dict()
                for neighbor in node_neighs[f_node]:
                    if neighbor in visited_nodes:
                        continue
                    # print('eval node: ', neighbor)
                    new_triples = drop_triples(subgraph_triples, neighbor)
                    if len(new_triples) == 0:
                        visited_nodes.append(neighbor)
                        # key_nodes.append(neighbor)
                        new_score_list[neighbor] = 1000000
                        break
                    score = importance_eval(new_triples, nentity, nrelation, args, kge_model, optimizer, outer_nodes, fixed_embed, h, r, t)
                    new_score_list[neighbor] = torch.abs(score - pre_score)
                    visited_nodes.append(neighbor)
                new_score_list_sorted = sorted(new_score_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
                topk_nodes=[]
                for i in range(args.top_k):
                    if i<len(new_score_list_sorted):
                        topk_nodes.append(new_score_list_sorted[i][0])
                
                key_nodes.extend(topk_nodes)
                for i_node in topk_nodes:
                    finding_nodes.insert(0, i_node)
                    # print(f'insert node {i_node} into the queue!')
                
            # print('finded key nodes: ', key_nodes)
            explainable_subgraph = dgl.node_subgraph(graph, key_nodes).to(args.device)
            pred_score, score_kge = eval_prediction(model, explainable_subgraph)
            
            pred_scores.append(pred_score)
            kge_scores.append(score_kge)
            bar.set_description(f'{pred_score.cpu().numpy()}/{score_kge.cpu().numpy()}')
    
    pre_scores = torch.concat(pred_scores)
    kge_scores = torch.concat(kge_scores)
    # print(pre_scores, kge_scores)
    print('saving scores!!!')
    np.save(f'datasets/{args.dataset}/model_explainable_scores_pre.npy', pre_scores.detach().cpu().numpy())
    np.save(f'datasets/{args.dataset}/model_explainable_scores_kge.npy', kge_scores.detach().cpu().numpy())

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args)

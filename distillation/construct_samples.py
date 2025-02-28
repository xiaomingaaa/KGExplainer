import argparse
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F

def get_true_head_and_tail(triples):
    '''
    Build a dictionary of true triples thatwill
    be used to filter these true triples fornegative sampling
    '''
    
    true_head = {}
    true_tail = {}
    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)
    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation,tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head,relation)])))                 
    return true_head, true_tail

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

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Negative sample for Distillation',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--dataset', type=str,default='wn18rr', help='used dataset')
    parser.add_argument('--n_size', type=int,default=3, help='negative sample size')
    
    return parser.parse_args(args)

def TransE(head, relation, tail, mode):
    if mode == 'head-batch':
        score = head + (relation - tail)
    else:
        score = (head + relation) - tail
    embed = score
    score = torch.Tensor(24) - torch.norm(score, p=1, dim=0)
    return score.mean()

def construct_samples(args, triples, nentity, nrelation, negative_sample_size, true_head, true_tail, mode='tail-batch'):
    total_samples = []
    entity_embed = torch.from_numpy(np.load(f'ckpts/TransE_{args.dataset}_0/entity_embedding.npy'))
    relation_embed = torch.from_numpy(np.load(f'ckpts/TransE_{args.dataset}_0/relation_embedding.npy'))
    for i in range(len(triples)):
        print(i)
        head, relation, tail = triples[i]
        # p_score = TransE(entity_embed[head], relation_embed[relation], entity_embed[tail], mode='single')
        # p_score = F.logsigmoid(p_score).numpy()
        total_samples.append([head, relation, tail, 0])

        ### head_batch
        negative_sample = np.random.randint(nentity, size=negative_sample_size)
        mask = np.in1d(
                negative_sample, 
                true_head[(relation,tail)], 
                assume_unique=True, 
                invert=True
            )
        
        negative_sample = negative_sample[mask]
        for i in range(len(negative_sample)):
            # score = TransE(entity_embed
            #                [negative_sample[i]], relation_embed[relation], entity_embed[tail], mode='head-batch')
            # score = F.logsigmoid(score)
            total_samples.append([negative_sample[i], relation, tail, 1])

        

        ### tail_batch
        negative_sample = np.random.randint(nentity, size=negative_sample_size)
        mask = np.in1d(
                    negative_sample, 
                    true_tail[(head,relation)], 
                    assume_unique=True, 
                    invert=True
                )
        negative_sample = negative_sample[mask]
        for i in range(len(negative_sample)):
            # score = TransE(entity_embed
            #                [head], relation_embed[relation], entity_embed[negative_sample[i]], mode='tail-batch')
            # score = F.logsigmoid(score)
            total_samples.append([head, relation, negative_sample[i], 0])

        # import ipdb;ipdb.set_trace()

    pd.DataFrame(total_samples).to_csv(f'data/{args.dataset}/distill_samples.txt',sep='\t',index=False, header=None)
    

if __name__=='__main__':
    args = parse_args()
    train_path = f'data/{args.dataset}/train.txt'
    with open(f'data/{args.dataset}/entities.dict') as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(f'data/{args.dataset}/relations.dict') as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triples = read_triple(train_path, entity2id, relation2id)

    true_head, true_tail = get_true_head_and_tail(train_triples)

    construct_samples(args, train_triples, len(entity2id), len(relation2id),args.n_size,true_head, true_tail)

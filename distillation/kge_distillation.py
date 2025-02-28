import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl import apply_each
from dgl.dataloading import DataLoader, NeighborSampler

class HeteroGAT(nn.Module):
    def __init__(self, etypes, num_nodes, in_size, hid_size, out_size, n_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.embed = nn.Embedding(num_nodes, in_size)
        
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(in_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(hid_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(hid_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.dropout = nn.Dropout(0.5)
        self.output_layers = nn.ModuleList()
        self.output_layers.append(
            nn.Linear(hid_size, out_size)
        )
        self.output_layers.append(
            nn.Linear(out_size, 1)
        )
        self.output_layers.append(nn.LogSigmoid())
        
    def forward(self, g):
        h = {'node':self.embed(g.ndata[dgl.NID])}
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            # One thing is that h might return tensors with zero rows if the number of dst nodes
            # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        g.ndata['feat'] = h['node']
        embed = dgl.sum_nodes(g, 'feat')
        score = embed
        for layer in self.output_layers:
            score = layer(score)
        return embed, score
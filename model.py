'''
Created on Oct 12, 2023
Pytorch Implementation of TempLGCN in
Tseesuren et al. tempLGCN: Temporal Collaborative Filtering with Graph Convolutional Networks

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

from torch import nn, Tensor
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from torch_geometric.utils import softmax
import torch.nn.functional as F

class LGCN(MessagePassing):    
    def __init__(self, 
                 model,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 num_layers=3,
                 add_self_loops = False,
                 mu = 0,
                 u_stats = None,
                 verbose = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.edge_index_norm = None
        self.verbose = verbose
        self.mu = mu
        self.model = model
        
        self.user_baseline = False
        self.item_baseline = False
        self.u_abs_drift = False
        self.u_rel_drift = False
        
        if model == 'lgcn_b':
            self.item_baseline = True
            self.user_baseline = True
        elif model == 'lgcn_b_a':
            self.u_abs_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif model == 'lgcn_b_r':
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif model == 'lgcn_b_ar':
            self.u_abs_drift = True
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        else:
            model = 'lgcn'
            self.mu = 0
        
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        
        self.users_emb.weight.requires_grad = True
        self.items_emb.weight.requires_grad = True
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        if self.user_baseline:
            self._u_base_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim)
            nn.init.zeros_(self._u_base_emb.weight)
            self._u_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The user baseline embedding is ON.")
        
        if self.item_baseline:
            self._i_base_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim)
            nn.init.zeros_(self._i_base_emb.weight)
            self._i_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The item baseline embedding is ON.")

        if self.u_abs_drift:
            self._u_abs_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim)     
            nn.init.zeros_(self._u_abs_drift_emb.weight)
            self._u_abs_drift_emb.weight.requires_grad = True
            if self.verbose:
                print("The absolute user drift temporal embedding is ON.")

        if self.u_rel_drift:
            self._u_rel_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim)     
            nn.init.zeros_(self._u_rel_drift_emb.weight)
            self._u_rel_drift_emb.weight.requires_grad = True
            if self.verbose:
                print("The relative user drift temporal embedding is ON.")
                
        self.f = nn.ReLU()
        
    def forward(self, src: Tensor, dest: Tensor, edge_index: Tensor, u_abs_t_decay: Tensor, u_rel_t_decay: Tensor):
        
        if(self.edge_index_norm is None):
            self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
                    
        u_emb_0 = self.users_emb.weight
        i_emb_0 = self.items_emb.weight
        emb_0 = torch.cat([u_emb_0, i_emb_0])
        embs = [emb_0]
        emb_k = emb_0
    
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index=self.edge_index_norm[0], x=emb_k, norm=self.edge_index_norm[1])
            embs.append(emb_k)
             
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)          
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]
        
        _inner_pro = torch.mul(user_embeds, item_embeds)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[src]
            _inner_pro = _inner_pro + _u_base_emb
        
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[dest]
            _inner_pro = _inner_pro + _i_base_emb
        
        if self.u_abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[src]
            _u_abs_drift_emb = _u_abs_drift_emb * u_abs_t_decay.unsqueeze(1)
            _inner_pro = _inner_pro + _u_abs_drift_emb
            
        if self.u_rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[src]
            _u_rel_drift_emb = _u_rel_drift_emb * u_rel_t_decay.unsqueeze(1) 
            _inner_pro = _inner_pro + _u_rel_drift_emb
                
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.model != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
        
        ratings = self.f(_inner_pro)
              
        return ratings
    
    def message(self, x_j, norm):
        out =  x_j * norm.view(-1, 1)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device
import numpy as np

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.a = nn.Linear(2 * in_features, 1)
        nn.init.xavier_uniform_(self.a.weight.data, gain=1.414)

        self.alpha = 0.02
        self.leaky_relu = nn.LeakyReLU(self.alpha)


    # def softmax(self, attention):
    #     exp_attention = torch.exp(attention)
    #     return exp_attention / torch.sum(exp_attention, dim=2)
    #
    #
    #
    # def forward(self, adj, hiddens):
    #     outputs = []
    #     indices = []
    #
    #     for a, h in zip(adj, hiddens):
    #         # print(h.shape)
    #         h_size = h.shape[0]
    #         if a is None:
    #             tmp_a = torch.eye(h_size)
    #             a_size = h_size
    #             tmp_h = h
    #
    #         else:
    #             # tmp_a = torch.zeros(h_size, h_size)
    #             if type(a) == np.ndarray:
    #                 a = torch.from_numpy(a)
    #             # print(tmp_adj.shape, a.shape)
    #             tmp_a = a
    #             # tmp_a[:a.shape[1], :a.shape[1]] += a
    #             a_size = a.shape[1]
    #             tmp_h = h[:a_size]
    #
    #         tmp_a = tmp_a.to(device)
    #
    #         Wh1 = torch.matmul(tmp_h, self.a.weight.T[:128, :])
    #         Wh2 = torch.matmul(tmp_h, self.a.weight.T[128:, :])
    #         e = Wh1 + Wh2.T
    #         e = self.leaky_relu(e)
    #
    #         zero_vec = -9e15 * torch.ones_like(e)
    #         attention_adj = torch.where(tmp_a != 0, e, zero_vec)
    #         attention = self.softmax(attention_adj)
    #
    #         h_prime = torch.matmul(attention, tmp_h).unsqueeze(0)
    #         hidden, idx = F.max_pool2d(h_prime, kernel_size=(attention.shape[1], 1), return_indices=True)
    #         outputs.append(hidden)
    #         indices.append(idx)
    #
    #     outputs = torch.cat(outputs, dim=0)
    #     indices = torch.cat(indices, dim=0)
    #     return outputs, indices




    def forward(self, adj, Wh):

        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 *torch.ones_like(e)
        tmp_adj = torch.zeros(Wh.shape[0], Wh.shape[1], Wh.shape[1]).to(device)

        # print(adj)

        for idx, a in enumerate(adj):
            if a is None :
                tmp_adj[idx, :Wh.shape[1], :Wh.shape[1]] = torch.eye(Wh.shape[1]).to(device)
            else:
                # print(a)
                if type(a) == np.ndarray:
                    a = torch.from_numpy(a)

                # print(a.shape, Wh.shape)
                tmp_adj[idx, :a.shape[1], :a.shape[1]] += a.to(device)

                # tmp_adj[idx, :a.shape[1], :a.shape[1]] += a.squeeze(0).to(device)

        # for idx, a in enumerate(adj):
        #
        #     tmp_e[idx, :, :] = e[idx, :, :]
        #     if a is None :
        #         tmp_adj[idx, :, :] = 1
        #         tmp_e[idx ,range(e.shape[1]) ,range(e.shape[1])] = 1
        #
        #     else:
        #
        #         tmp_adj[idx, :a.shape[1], :a.shape[1]] += a.squeeze(0)
        attention = torch.where(tmp_adj != 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, Wh)
        # h_prime = F.layer_norm(h_prime, h_prime.shape[2:])
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # print(Wh.shape, self.a.shape)
        Wh1 = torch.matmul(Wh, self.a.weight.T[:self.in_features, :])
        Wh2 = torch.matmul(Wh, self.a.weight.T[self.in_features:, :])
        # Wh1 = torch.matmul(Wh, self.a[:self.in_features, :])
        # Wh2 = torch.matmul(Wh, self.a[self.in_features:, :])
        e = Wh1 + Wh2.transpose(2 ,1)


        return self.leaky_relu(e)

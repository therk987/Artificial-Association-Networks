import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.option import device



class DeconvGatedRecurrentUnit(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim



        self.X2H = nn.Parameter(torch.empty(size=(input_dim, 3 * hidden_dim)))
        nn.init.xavier_uniform_(self.X2H.data, gain=1.414)

        self.H2H = nn.Parameter(torch.empty(size=(hidden_dim, 3 * hidden_dim)))
        nn.init.xavier_uniform_(self.H2H.data, gain=1.414)


        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()




    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)



    def gated_inputs(self, x, hidden):

        gate_x = torch.matmul(x, self.X2H)
        gate_h = torch.matmul(hidden, self.H2H)
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)

        irh = i_n + (resetgate * h_n)
        irh = F.layer_norm(irh, irh.shape[2:])
        newgate = F.tanh(irh)
        #         newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, adj, x, h):
        Wh = self.gated_inputs(x, h)

        zero_vec = -9e15 *torch.ones_like(e)
        tmp_adj = torch.zeros(Wh.shape[0], Wh.shape[1], Wh.shape[1]).to(device)
        for idx, a in enumerate(adj):
            if a is None :
                tmp_adj[idx, :Wh.shape[1], :Wh.shape[1]] = torch.eye(Wh.shape[1]).to(device)
            else:
                tmp_adj[idx, :a.shape[1], :a.shape[1]] += a.squeeze(0)

        conv_adj = self.prepare_spectral_graph_inputs(tmp_adj)
        h_prime = torch.matmul(conv_adj, Wh)
        return h_prime, Wh




    def prepare_spectral_graph_inputs(self, graphs):
        d_graph = torch.unsqueeze(torch.sum(graphs ,2) ,1) * torch.eye(graphs.shape[1]).to(device)
        r_graph = torch.inverse(d_graph ** 0.5)
        adj = torch.matmul(torch.matmul(r_graph , graphs) , r_graph)
        return adj

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.hidden_dim) + ')'


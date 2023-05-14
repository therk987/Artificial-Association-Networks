import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device

class MaxpoolReadoutLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, child_counts):

        node_count = hidden.shape[1]

        tmp_hidden = torch.ones_like(hidden).to(device) * -99999
        for i, c_count in enumerate(child_counts):

            if c_count == 0:
                tmp_hidden[i][:1] = hidden[i][:1]
            else:
            # if c_count > 0 and node_count - c_count > 0 :
                tmp_hidden[i][:c_count] = hidden[i][:c_count]
            # if c_count > 0 and node_count - c_count > 0 :
            #     hidden[i][c_count:] = -100000
            # print('ORIG',c_count, hidden[i][:c_count])
            #
            # print('NULL ITEM',c_count, hidden[i][c_count:])
            # print(c_count, hidden[i][c_count:].shape)
            # hidden, indices = F.max_pool2d(hidden[i][c_count:].unsqueeze(0), kernel_size=(c_count, 1), return_indices=True)


        tmp_hidden = tmp_hidden.unsqueeze(1)
        tmp_hidden, indices = F.max_pool2d(tmp_hidden, kernel_size=(node_count, 1), return_indices=True)
        tmp_hidden = tmp_hidden.squeeze(1)
        return tmp_hidden, indices


        # hidden = hidden.unsqueeze(1)
        # hidden, indices = F.max_pool2d(hidden, kernel_size=(node_count, 1), return_indices=True)
        # hidden = hidden.squeeze(1)
        # return hidden, indices

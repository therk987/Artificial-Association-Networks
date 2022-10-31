import torch
from torch.nn import functional as F
import torch.nn as nn
import pickle

from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from representation.nlp_analysis import utils
from config.option import device
import os

class ConvolutionalSentimentNetworks(nn.Module):

    def __init__(self, TEXT):
        super().__init__()
        self.TEXT = TEXT
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        N_FILTERS = 100
        FILTER_SIZES = [3, 4, 5]
        OUTPUT_DIM = 128
        DROPOUT = 0.0
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        n_filters = N_FILTERS
        filter_sizes = FILTER_SIZES
        output_dim = OUTPUT_DIM
        dropout = DROPOUT


        glove_embedding = torch.FloatTensor(TEXT.vocab.vectors)
        self.embedding = nn.Embedding.from_pretrained(
            glove_embedding,
            padding_idx=PAD_IDX,
            freeze=True).to(device)

        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        self.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        self.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]




        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], EMBEDDING_DIM))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], EMBEDDING_DIM))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], EMBEDDING_DIM))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.parameter_init()

    def parameter_init(self):
        params = [self.conv_0, self.conv_1, self.conv_2, self.fc]
        for param in params:
            nn.init.xavier_uniform_(param.weight.data, gain=1.414)


    def forward(self, batch_tree:BatchNeuroTree):
        batch_x = batch_tree.get('x')
        # x = torch.stack(batch_x).to(device)

        text_matrix = self.TEXT.process(batch_x).to(device)

        # text = [batch size, sent len]

        #         print(text)
        embedded = self.embedding(text_matrix)
        #         print(embedded.shape)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]
        out = self.fc(cat)

        # print(out.shape)
        return out

    def setting_pretrained_model(self, model):
        self.conv_0 = model.conv_0
        self.conv_1 = model.conv_1
        self.conv_2 = model.conv_2

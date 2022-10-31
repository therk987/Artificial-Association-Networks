import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device


class ConvolutionalSentimentNetworks(nn.Module):
    def __init__(self, TEXT):
        super().__init__()

        self.TEXT = TEXT
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        N_FILTERS = 100
        FILTER_SIZES = [3, 4, 5]
        OUTPUT_DIM = 2
        DROPOUT = 0.0
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]


        vocab_size = INPUT_DIM
        embedding_dim = EMBEDDING_DIM
        n_filters = N_FILTERS
        filter_sizes = FILTER_SIZES
        dropout = DROPOUT

        glove_embedding = torch.FloatTensor(TEXT.vocab.vectors)
        self.embedding = nn.Embedding.from_pretrained(
            glove_embedding,
            padding_idx=PAD_IDX,
            freeze=True).to(device)

        self.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        self.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, OUTPUT_DIM)

        self.dropout = nn.Dropout(dropout)

        self.parameter_init()

    def parameter_init(self):
        params = [self.conv_0, self.conv_1, self.conv_2, self.fc]
        for param in params:
            nn.init.xavier_uniform_(param.weight.data, gain=1.414)



    def forward(self, text):

        text = self.TEXT.process(text).to(device)

        # text = [batch size, sent len]

        #         print(text)
        embedded = self.embedding(text)
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

        out = self.fc(cat)
        # cat = [batch size, n_filters * len(filter_sizes)]

        return F.log_softmax(out, dim = -1)
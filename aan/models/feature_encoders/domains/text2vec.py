import torch
from torch import nn
import torch.nn.functional as F

from aan.data_structures.batch_neurotree import BatchNeuroTree


class ConvolutionalSentimentNetworks(nn.Module):
    """Text feature extractor psi_text based on Kim 2014 CNN (paper, Table XII).

    ``TEXT`` is a torchtext(-legacy) Field carrying the vocabulary and GloVe
    vectors; it is injected so this module stays torchtext-version agnostic.
    Any object exposing ``vocab.vectors``, ``vocab.stoi``, ``pad_token``,
    ``unk_token`` and ``process(list_of_token_lists) -> LongTensor`` works.
    """

    def __init__(self, TEXT, output_dim=128,
                 embedding_dim=100, n_filters=100, filter_sizes=(3, 4, 5), dropout=0.0):
        super().__init__()
        self.TEXT = TEXT

        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        unk_idx = TEXT.vocab.stoi[TEXT.unk_token]

        glove_embedding = torch.FloatTensor(TEXT.vocab.vectors)
        self.embedding = nn.Embedding.from_pretrained(
            glove_embedding, padding_idx=pad_idx, freeze=True)
        self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
        self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.parameter_init()

    def parameter_init(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.fc.weight.data, gain=1.414)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.fc.weight.device
        batch_x = batch_tree.getX()
        text_matrix = self.TEXT.process(batch_x).to(device)

        embedded = self.embedding(text_matrix).unsqueeze(1)
        # embedded: (batch, 1, sent len, emb dim)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

    def setting_pretrained_model(self, model):
        self.convs = model.convs


class TextCNNEncoder(nn.Module):
    """torchtext-free psi_text: the same Kim-2014 CNN over pre-tokenized id
    tensors stored on the leaf nodes, with a trainable embedding.

    (The paper's version used frozen GloVe vectors via torchtext; torchtext
    is deprecated upstream, so the revision experiments train the embedding
    from scratch — note this when comparing against the original numbers.)
    """

    def __init__(self, vocab_size, output_dim=128,
                 embedding_dim=100, n_filters=100, filter_sizes=(3, 4, 5),
                 dropout=0.0, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.fc.weight.data, gain=1.414)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.fc.weight.device
        text_matrix = torch.stack(batch_tree.getX()).to(device)  # (B, L) long

        embedded = self.embedding(text_matrix).unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

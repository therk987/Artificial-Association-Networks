from torch.nn import functional as F
import torch.nn as nn
import torch
import pickle

from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from representation.nlp_analysis import utils
from config.option import device
import os





class Word2Vec(nn.Module):
    def __init__(self, model_name):

        super().__init__()

        self.model_name = model_name
        self.embedding, self.vocab = self.setting_embedding(model_name)

        self.fc = nn.Linear(300, 128)

        self.parameter_init()

    def setting_embedding(self, model_name):
        if model_name == 'glove':
            MODEL_HOME = 'models/pretrained'
            GLOVE_HOME = os.path.join(MODEL_HOME, 'glove.6B')


            with open(MODEL_HOME + '/sst_train_vocab.pkl', 'rb') as f:
                sst_train_vocab = pickle.load(f)

            lookup = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))
            embedding, vocab = utils.create_pretrained_embedding(
                lookup, sst_train_vocab)

            embedding = torch.FloatTensor(embedding)
            pretrained_embedding = nn.Embedding.from_pretrained(
                embedding, freeze=True).to(device)


        return pretrained_embedding, vocab

    def parameter_init(self):

        #     def _reset_params(self):

        params = [self.fc]
        for param in params:
            nn.init.xavier_uniform_(param.weight.data, gain=1.414)


    def forward(self, batch_tree:BatchNeuroTree):
        batch_x = batch_tree.get('x')
        x = torch.stack(batch_x).to(device)
        out = self.embedding(x)
        out = self.fc(out)
        out = F.normalize(out)
        out = F.relu(out)
        return out.squeeze(1)




class Word2Vec_IMDB(nn.Module):
    def __init__(self, TEXT):
        super().__init__()

        self.TEXT = TEXT

        self.fc = nn.Linear(300, 128)

        self.parameter_init()
    #
    # def setting_embedding(self, model_name):
    #     if model_name == 'glove':
    #         MODEL_HOME = 'models/pretrained'
    #         GLOVE_HOME = os.path.join(MODEL_HOME, 'glove.6B')
    #
    #
    #         with open(MODEL_HOME + '/sst_train_vocab.pkl', 'rb') as f:
    #             sst_train_vocab = pickle.load(f)
    #
    #         lookup = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))
    #         embedding, vocab = utils.create_pretrained_embedding(
    #             lookup, sst_train_vocab)
    #
    #         embedding = torch.FloatTensor(embedding)
    #         pretrained_embedding = nn.Embedding.from_pretrained(
    #             embedding, freeze=True).to(device)
    #
    #
    #     return pretrained_embedding, vocab

    def parameter_init(self):

        #     def _reset_params(self):

        params = [self.fc]
        for param in params:
            nn.init.xavier_uniform_(param.weight.data, gain=1.414)


    def forward(self, batch_tree:BatchNeuroTree):
        batch_x = batch_tree.get('x')
        text_matrix = self.TEXT.process(batch_x).to(device)
        # text = [batch size, sent len]
        #         print(text)
        embedded = self.embedding(text_matrix)
        #         print(embedded.shape)
        # embedded = [batch size, sent len, emb dim]

        out = self.fc(embedded)
        out = F.normalize(out)
        out = F.relu(out)
        return out.squeeze(1)


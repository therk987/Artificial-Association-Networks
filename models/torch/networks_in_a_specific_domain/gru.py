import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from config.option import device

class GRU_for_mfcc(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.fc_in = nn.Linear(40, 128)
        self.rnn = nn.GRU(128, 128, num_layers=1, batch_first=True)
        self.fc_layer = nn.Linear(128, 35)

        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate).to(device)

        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, data):
        sound_features = self.mfcc_transform(data)
        sound_features = sound_features.squeeze(1).transpose(1, 2)

        x = self.fc_in(sound_features)
        x = F.normalize(x)
        x = F.relu(x)
        x, h = self.rnn(x)
        x = self.fc_layer(h).squeeze(0)
        x = F.log_softmax(x, dim=-1)
        return x


class GRU_for_imdb(nn.Module):
    def __init__(self, TEXT):
        super().__init__()

        self.TEXT = TEXT
        self.fc_in = nn.Linear(300, 128)
        self.rnn = nn.GRU(128, 128, num_layers=1, batch_first=True)
        self.fc_layer = nn.Linear(128, 35)

        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, x):

        inputs = self.TEXT.process(x)


        x = self.fc_in(inputs.squeeze(1).transpose(2, 1))
        x = F.normalize(x)
        x = F.relu(x)
        x, h = self.rnn(x)
        x = self.fc_layer(h).squeeze(0)
        x = F.log_softmax(x, dim=-1)
        return x

from torch import nn
import torch.nn.functional as F
import torchaudio

class GroupNorm(nn.Module):
    def __init__(self, tmp):
        super(GroupNorm, self).__init__()

    def forward(self, inputs):
#         return F.layer_norm(inputs, inputs.shape[1:])
        return F.group_norm(inputs, 16)


sample_rate = 16000
new_sample_rate = 8000
# waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

class M5_GroupNorm(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)  # (in, out, filter size, stride)
        self.bn1 = GroupNorm(128)  # normalize
        #         self.bn1 = nn.LayerNorm()
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)  # by default,the stride is 1 if it is not specified here.
        self.bn2 = GroupNorm(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = GroupNorm(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = GroupNorm(512)
        self.pool4 = nn.MaxPool1d(4)

        self.avgPool = nn.AdaptiveAvgPool1d(
            1)  # insteads of using nn.AvgPool1d(30) (where I need to manually check the dimension that comes in). I use adaptive n flatten
        # the advantage of adaptiveavgpool is that it manually adjust to avoid dimension issues
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 35)  # this is the output layer.

        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)



    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        # print(x.shape)
        x = self.avgPool(x)
        x = self.flatten(x)  # replaces permute(0,2,1) with flatten
        x = self.fc1(x)  # output layer ([n,1, 10] i.e 10 probs. for each audio files)
        return F.log_softmax(x, dim=1)  # we didnt use softmax here becuz we already have that in cross entropy



from torch.nn import functional as F
import torch.nn as nn
import torch
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from config.option import device

class GroupNorm(nn.Module):
    def __init__(self, tmp):
        super(GroupNorm, self).__init__()

    def forward(self, inputs):
#         return F.layer_norm(inputs, inputs.shape[1:])
        return F.group_norm(inputs, 16)

class M5_GroupNorm(nn.Module):                           # this is m5 architecture
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)  #(in, out, filter size, stride)
        self.bn1 = GroupNorm(128)         #normalize 
#         self.bn1 = nn.LayerNorm()
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)    #by default,the stride is 1 if it is not specified here.
        self.bn2 = GroupNorm(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = GroupNorm(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = GroupNorm(512)
        self.pool4 = nn.MaxPool1d(4)
        
        self.avgPool = nn.AdaptiveAvgPool1d(1) #insteads of using nn.AvgPool1d(30) (where I need to manually check the dimension that comes in). I use adaptive n flatten
        #the advantage of adaptiveavgpool is that it manually adjust to avoid dimension issues
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, output_size)          # this is the output layer.
        
        self.parameter_init()
    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

        
    def getParameters(self):
        all_network = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1]

        parameters = [
                        ]

        for net in all_network:
            parameters.extend(net.parameters())

        print('M5.getParameters()', len(parameters))

            
        return parameters   
                       
        
    def forward(self, batch_tree:BatchNeuroTree):
        batch_x = batch_tree.get('x')
        x = torch.stack(batch_x).to(device)

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
        x = self.fc1(x)       #output layer ([n,1, 10] i.e 10 probs. for each audio files) 
        return F.leaky_relu(x) # we didnt use softmax here becuz we already have that in cross entropy


    def setting_pretrained_model(self, model):
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv3 = model.conv3
        self.conv4 = model.conv4

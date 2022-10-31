import torch.nn as nn
import torch.nn.functional as F
from models.torch.networks.association_networks.recursive.conv import RecursiveAssociationNeuralNetworks
from models.torch.networks.feature_extraction_networks.text.word2vec import Word2Vec
from models.torch.networks.task_networks.classification import Classification

from models.torch.networks.multi_connection_networks.multi_encoder import MultiExtractionConnector
from models.torch.networks.multi_connection_networks.multi_subtask import MultiSubTaskConnector
from models.torch.networks.multi_connection_networks.multi_maintask import MultiMainTaskConnector
import torch
from models.torch.dataloader.batch_neurotree import BatchNeuroTree

from config.option import device


class RecursiveNeuralNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_feature_extraction_networks = self.create_multi_feature_extraction_networks()
        self.multi_sub_task_networks = self.create_multi_subtask_networks()
        self.ran = RecursiveAssociationNeuralNetworks(128, 128, 1, self.multi_feature_extraction_networks,
                                                      self.multi_sub_task_networks, version='ran')
        self.multi_main_task_networks = self.create_multi_maintask_networks()

        self.classification = nn.Linear(128, 2)


    def create_multi_feature_extraction_networks(self):
        self.word2vec = Word2Vec('glove').to(device)

        fe_networks = {
            'language': self.word2vec
        }
        return MultiExtractionConnector(128, fe_networks).to(device)

    def create_multi_subtask_networks(self):
        self.sub_classification = Classification(128, 10).to(device)
        task_networks = {
            'classification': self.sub_classification
        }
        return MultiSubTaskConnector(128, task_networks).to(device)

    def create_multi_maintask_networks(self):
        # 62
        self.main_classification = Classification(128, 2).to(device)
        task_networks = {
            'classification': self.main_classification,
        }
        return MultiMainTaskConnector(128, task_networks).to(device)

    def forward(self, inputs, node_level = False):
        batchNeuroTree = BatchNeuroTree(inputs)
        h_root, batchNeuroTree = self.propagate(batchNeuroTree, node_level)
        y = self.classification(h_root)
        y = F.log_softmax(y, dim = -1)
        return y

    def propagate(self, batchNeuroTree, node_level = False):
        outputs = self.ran(batchNeuroTree, node_level)
        return torch.stack(outputs, dim=0), batchNeuroTree

import torch.nn as nn
import torch.nn.functional as F
from models.torch.networks.association_networks.recursive.conv import RecursiveAssociationNeuralNetworks
from models.torch.networks.task_networks.classification import Classification
from models.torch.networks.feature_extraction_networks.code.ast2vec_for_python import Ast2VectorForPython
from models.torch.networks.feature_extraction_networks.code.num2vec import Num2Vec

from models.torch.networks.multi_connection_networks.multi_encoder import MultiExtractionConnector
from models.torch.networks.multi_connection_networks.multi_subtask import MultiSubTaskConnector
from models.torch.networks.multi_connection_networks.multi_maintask import MultiMainTaskConnector
import torch
from models.torch.dataloader.batch_neurotree import BatchNeuroTree

from config.option import device


class CodeRecursiveNeuralNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_feature_extraction_networks = self.create_multi_feature_extraction_networks()
        self.multi_sub_task_networks = self.create_multi_subtask_networks()
        self.ran = RecursiveAssociationNeuralNetworks(128, 128, 1, self.multi_feature_extraction_networks,
                                                      self.multi_sub_task_networks, version='ran')
        self.multi_main_task_networks = self.create_multi_maintask_networks()
        self.classification = nn.Linear(128, 6)

    def create_multi_feature_extraction_networks(self):
        self.astcls2vec = Ast2VectorForPython(128).to(device)
        self.astnum2vec = Num2Vec(128).to(device)

        fe_networks = {
            'ast_class': self.astcls2vec,
            'ast_num': self.astnum2vec,
        }
        return MultiExtractionConnector(128, fe_networks).to(device)

    def create_multi_subtask_networks(self):
        self.sub_classification = Classification(128, 70).to(device)
        task_networks = {
            'classification': self.sub_classification
        }
        return MultiSubTaskConnector(128, task_networks).to(device)

    def create_multi_maintask_networks(self):
        # 62
        self.main_classification = Classification(128, 6).to(device)
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

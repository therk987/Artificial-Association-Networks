import torch.nn as nn

from models.encoders.recursive_encoder import RecursiveAssociationNeuralNetworks
from models.feature_encoders.multiencoder_connector import MultiExtractionConnector
from models.feature_decoders.multidecoder_connector import MultiRestorationConnector

from models.subtasks.multisubtask_connector import MultiSubTaskConnector
from models.maintasks.multimaintask_connector import MultiMainTaskConnector
import torch

from config.option import device






class ArtificialAssociationNeuralNetworks(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            feature_encoders:dict, feature_decoders:dict, subtask_networks:dict, maintask_networks:dict,
            version = 'gaau',
            device = device
            ):
        super(ArtificialAssociationNeuralNetworks, self).__init__()
        self.multi_feature_extraction_networks = self.create_multi_feature_extraction_networks(feature_encoders, device)
        self.multi_restoration_networks = self.create_multi_restoration_networks(feature_decoders, device)
        self.multi_sub_task_networks = self.create_multi_subtask_networks(subtask_networks, device)


        if version == 'ran':
            self.ran = RecursiveAssociationNeuralNetworks(input_dim, hidden_dim, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)

        elif version == 'raan':
            self.ran = RecursiveAssociationNeuralNetworks(input_dim, hidden_dim, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)


        elif version == 'gau':
            self.ran = RecursiveAssociationNeuralNetworks(input_dim, hidden_dim, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)

        elif version == 'gaau':
            self.ran = RecursiveAssociationNeuralNetworks(input_dim, hidden_dim, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)

        elif version == 'egaau':
            self.ran = RecursiveAssociationNeuralNetworks(input_dim, hidden_dim, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)
        else:
            assert False

        self.multi_main_task_networks = self.create_multi_maintask_networks(maintask_networks, device)

    def create_multi_feature_extraction_networks(self, feature_encoders : dict, device):
        return MultiExtractionConnector(128, feature_encoders).to(device)

    def create_multi_restoration_networks(self, feature_decoders: dict, device):
        return MultiRestorationConnector(128, feature_decoders).to(device)

    def create_multi_subtask_networks(self, subtask_networks: dict, device):
        return MultiSubTaskConnector(128, subtask_networks).to(device)

    def create_multi_maintask_networks(self, maintask_networks : dict, device):
        return MultiMainTaskConnector(128, maintask_networks).to(device)


    def forward(self, batchNeuroTree, tasks, node_level = False):
        h_root, batchNeuroTree = self.propagate(batchNeuroTree, node_level)
        main_task_outputs = self.multi_main_task_networks(h_root, batchNeuroTree, tasks)
        return main_task_outputs, h_root, batchNeuroTree


    def propagate(self, batchNeuroTree, node_level = False):
        # current1 = time.time()
        outputs = self.ran(batchNeuroTree, node_level)
        # current2 = time.time()
        # print('time 1', current2 - current1)
        # self.dran(batchNeuroTree.get_task_nodes('autoencoder'))
        # current3 = time.time()
        # print('time 2',current3-current2)
        return torch.stack(outputs, dim=0), batchNeuroTree
        # if node_level:
        #     return batchNeuroTree
        # else:
        #     return torch.stack(outputs, dim = 0)

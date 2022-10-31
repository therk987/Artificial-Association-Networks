import torch.nn as nn
from models.torch.networks.association_networks.level.conv import LevelAssociationNeuralNetworks

from models.torch.networks.association_networks.recursive.conv import RecursiveAssociationNeuralNetworks
from models.torch.networks.association_networks.recursive.deconv import DeconvolutionalRecursiveAssociationNetworks
from models.torch.networks.feature_extraction_networks.image.image2vec import LeNet_5
from models.torch.networks.feature_extraction_networks.code.num2vec import Num2Vec
from models.torch.networks.feature_extraction_networks.sound.sound2vec import M5_GroupNorm
from models.torch.networks.feature_extraction_networks.code.ast2vec_for_java import Ast2VectorForJava
from models.torch.networks.feature_extraction_networks.tabular.tabular2vec import Tabular2Vec
from models.torch.networks.feature_extraction_networks.text.word2vec import Word2Vec
from models.torch.networks.feature_extraction_networks.graph.graph2vec import Graph2Vec
from models.torch.networks.feature_extraction_networks.text.cnn3 import ConvolutionalSentimentNetworks


from models.torch.networks.feature_restoration_networks.vec2instruction import DecAst2Vector
from models.torch.networks.feature_restoration_networks.vec2num import Vec2Num
from models.torch.networks.feature_restoration_networks.vec2image import Deconv_LeNet_5

from models.torch.networks.task_networks.classification import Classification
from models.torch.networks.task_networks.autoencoder import AutoEncoder

from models.torch.networks.multi_connection_networks.multi_encoder import MultiExtractionConnector
# from models.torch.networks.multi_connection_networks.multi_task import MultiTaskConnector
from models.torch.networks.multi_connection_networks.multi_subtask import MultiSubTaskConnector
from models.torch.networks.multi_connection_networks.multi_maintask import MultiMainTaskConnector

from models.torch.networks.multi_connection_networks.multi_decoder import MultiRestorationConnector
import time
import torch

from models.torch.dataloader.neurotree_builder import image
from models.torch.dataloader.batch_neurotree import BatchNeuroTree

from config.option import device


class ArtificialAssociationNeuralNetworks(nn.Module):
    def __init__(self, version = 'gaau', max_lv = 3):
        super(ArtificialAssociationNeuralNetworks, self).__init__()
        self.multi_feature_extraction_networks = self.create_multi_feature_extraction_networks()
        # self.multi_restoration_networks = self.create_multi_restoration_networks()
        self.multi_sub_task_networks = self.create_multi_subtask_networks()

        # if version == 'gaau':
        #     self.dran = DeconvolutionalRecursiveAssociationNetworks(128, 128, self.multi_restoration_networks,
        #                                                             version=version)
        # elif version == 'lan':
        #     self.dran = DeconvolutionalRecursiveAssociationNetworks(128, 128, self.multi_restoration_networks,
        #                                                             version=version)
        # elif version == 'ran':
        #     self.dran = DeconvolutionalRecursiveAssociationNetworks(128, 128, self.multi_restoration_networks,
        #                                                             version=version)
        # else:
        #     assert False


        if version == 'lan':
            self.ran = LevelAssociationNeuralNetworks(128, 128, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version, max_lv = max_lv)

        elif version == 'laan':
            self.ran = LevelAssociationNeuralNetworks(128, 128, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version, max_lv = max_lv)


        elif version == 'ran':
            self.ran = RecursiveAssociationNeuralNetworks(128, 128, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)

        elif version == 'raan':
            self.ran = RecursiveAssociationNeuralNetworks(128, 128, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)


        elif version == 'gau':
            self.ran = RecursiveAssociationNeuralNetworks(128, 128, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)

        elif version == 'gaau':
            self.ran = RecursiveAssociationNeuralNetworks(128, 128, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)

        elif version == 'egaau':
            self.ran = RecursiveAssociationNeuralNetworks(128, 128, self.multi_feature_extraction_networks,
                                                          self.multi_sub_task_networks, version = version)
        else:
            assert False

        self.multi_main_task_networks = self.create_multi_maintask_networks()

    # def parameters(self):
    #     parameters = self.parameters()
    #     fe_parameters = self.multi_feature_extraction_networks.parameters()
    #     re_parameters = self.multi_restoration_networks.parameters()
    #     mt_parameters = self.multi_task_networks.parameters()
    #

    def create_multi_feature_extraction_networks(self):
        self.image2vec = LeNet_5().to(device)
        self.sound2vec = M5_GroupNorm(128).to(device)
        self.word2vec = Word2Vec('glove').to(device)
        self.graph2vec = Graph2Vec(128).to(device)
        self.tabular2vec = Tabular2Vec(128).to(device)
        self.astcls2vec = Ast2VectorForJava(128).to(device)
        self.astnum2vec = Num2Vec(128).to(device)

        fe_networks = {
            'image': self.image2vec,
            'sound': self.sound2vec,
            'language': self.word2vec,
            'graph': self.graph2vec,
            'tabular': self.tabular2vec,
            'ast_class': self.astcls2vec,
            'ast_num': self.astnum2vec,
        }
        return MultiExtractionConnector(128, fe_networks).to(device)

    def create_multi_restoration_networks(self):
        self.vec2image = Deconv_LeNet_5().to(device)
        self.vec2astcls = DecAst2Vector(128).to(device)
        self.vec2astnum = Vec2Num(128).to(device)

        fe_networks = {
            'image': self.vec2image,
            'sound': self.vec2sound,
            'language': self.vec2word,
            'graph': self.vec2graph,
            'tabular': self.vec2tabular,
            'ast_class': self.vec2astcls,
            'ast_num': self.vec2astnum,
        }

        return MultiRestorationConnector(128, fe_networks).to(device)

    def create_multi_subtask_networks(self):
        self.sub_classification = Classification(128, 10).to(device)
        task_networks = {
            'classification': self.sub_classification
        }
        return MultiSubTaskConnector(128, task_networks).to(device)

    def create_multi_maintask_networks(self):
        # 62
        self.main_classification = Classification(128, 62).to(device)
        # self.main_autoencoder = AutoEncoder(128, self.dran).to(device)
        task_networks = {
            'classification': self.main_classification,
            # 'autoencoder' : self.main_autoencoder
        }
        return MultiMainTaskConnector(128, task_networks).to(device)


    # def forward(self, inputs, node_level = False):
    #
    #     x = inputs
    #     treeset = []
    #     for tree in x:
    #         neuronode = image.buildImageGT(tree, 'classification')
    #         treeset.append(neuronode)
    #     batch_tree = BatchNeuroTree(treeset)
    #
    #     outputs = self.ran(batch_tree, node_level)
    #
    #     # main_tasks = batch_tree.get_task_nodes('autoencoder')
    #     # self.dran(batch_tree.get_task_nodes('autoencoder'))
    #     if node_level:
    #         return batch_tree
    #     else:
    #         return torch.stack(outputs, dim = 0)


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

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import pickle
import os
from dataset.image.load import MNIST_DATA
from dataset.sound.load import SPEECH_COMMAND_DATA
from dataset.text.load import SST_DATA
from dataset.text.load import IMDB_DATA
from dataset.graph.load import MUTAG_DATA

from dataset.graph.load import UPFD_DATA
from dataset.tabular.load import IRIS_DATA
from dataset.tabular.load import BREAST_CANCER_DATA
from dataset.code.load import ALGORITHM_AST_DATA
from dataset.code.load import ALGORITHM_NEUROTREE_DATA
from dataset.code.load import ALGORITHM_AST_NEUROTREE_DATA
from dataset.code.load import ALGORITHM_AST_NEUROTREE_FUNC_CALL_DATA
from dataset.code.load import SORT_AST_NEUROTREE_CONTROL_FLOW_GRAPH_DATA

from dataset.code.load import SORTING_CODECLONE_DATA
from dataset.code.load import BOJ_2110_DATA

def load_mnist_data():
    return MNIST_DATA()

def load_iris_data():
    return IRIS_DATA()

def load_speech_command_data():
    return SPEECH_COMMAND_DATA()

def load_sst_data():
    return SST_DATA()

def load_imdb_data():
    return IMDB_DATA()

def load_upfd_data():
    return UPFD_DATA()

def load_mutag_data():
    return MUTAG_DATA()

def load_sorting_codeclone_data():
    return SORTING_CODECLONE_DATA()

def load_algorithm_neurotree_data():
    return ALGORITHM_NEUROTREE_DATA()

def load_algorithm_ast_neurotree_func_call_data():
    return ALGORITHM_AST_NEUROTREE_FUNC_CALL_DATA()

def load_sort_ast_neurotree_control_flow_graph_data():
    return SORT_AST_NEUROTREE_CONTROL_FLOW_GRAPH_DATA()

def load_algorithm_ast_data():
    return ALGORITHM_AST_DATA()

def load_algorithm_ast_neurotree_data():
    return ALGORITHM_AST_NEUROTREE_DATA()


# load_algorithm_ast_neurotree_data()
#
# load_algorithm_data()
# load_mnist_data()
# load_speech_command_data()
# load_sst_data()
# load_upfd_data()
#


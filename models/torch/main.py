from models.torch.networks.artificial_association_networks import ArtificialAssociationNeuralNetworks
from models.torch.learning.config import TrainConfigure
from models.torch.dataloader.neurodataloader import createNeuroDataloader
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from models.torch.learning import train
from models.torch.learning import valid
import torch
from dataset.read import load_iris_data
from dataset.read import load_mnist_data
from dataset.read import load_speech_command_data
from dataset.read import load_sst_data
from dataset.read import load_upfd_data
from dataset.read import load_sorting_codeclone_data
from dataset.read import load_algorithm_data


from config.option import device


print('Device : ', device)




iris_train, iris_test = load_iris_data() # tabular
mnist_train, mnist_test = load_mnist_data() # image
speech_train, speech_test = load_speech_command_data() # sequences
sst_train, sst_test = load_sst_data() # tree
upfd_train, upfd_test = load_upfd_data() # graph
code_train, code_test = load_algorithm_data() # neurotree


neuro_x_train = {
    'tabular': iris_train.data.unsqueeze(1),
    'image': mnist_train.data.unsqueeze(1),
    'sound': speech_train.data.unsqueeze(1),
    'text': sst_train.data.unsqueeze(1),
    'graph': upfd_train.data.unsqueeze(1),
    'tree': upfd_train.data.unsqueeze(1),
    'neurotree': code_train.data.unsqueeze(1),
    # 'class': boj_2110_dataset
}

neuro_y_train = {
    'tabular': iris_test.data.unsqueeze(1),
    'image': mnist_test.data.unsqueeze(1),
    'sound': speech_test.data.unsqueeze(1),
    'text': sst_test.data.unsqueeze(1),
    'graph': upfd_test.data.unsqueeze(1),
    'tree': upfd_test.data.unsqueeze(1),
    'neurotree': code_test.data.unsqueeze(1),
    # 'class': [None for _ in range(len(boj_2110_dataset))]
}

neuro_x_test = {
    'image': mnist_test.data.unsqueeze(1),
    # 'class': boj_2110_dataset
}
neuro_y_test = {
    'image': mnist_test.targets,
    # 'class': [None for _ in range(len(boj_2110_dataset))]
}

# trainNeuroDataset, neuroloader_train = createNeuroDataloader(neuro_x_train, neuro_y_train,
#                                           {'image':'autoencoder',
#                                            'class': 'autoencoder'},  256, True)
# testNeuroDataset, neuroloader_test = createNeuroDataloader(neuro_x_test, neuro_y_test, {'image':'autoencoder', 'class': 'autoencoder'}, 32, False)

trainNeuroDataset, neuroloader_train = createNeuroDataloader(neuro_x_train, neuro_y_train,
                                            {'image':'classification'},
                                            {'image':'classification'},
                                                             256, True)
testNeuroDataset, neuroloader_test = createNeuroDataloader(neuro_x_test, neuro_y_test,
                                                           {'image': 'classification'},
                                                           {'image': 'classification'},
                                                             32, False)



model = ArtificialAssociationNeuralNetworks(version='gaau').to(device)
config = TrainConfigure(model,
                        model.multi_sub_task_networks,
                        model.multi_main_task_networks,
                        optimizer='adam',
                        lr = 0.01
)

# dummy_data = torch.empty(10, 1, 28, 28, dtype = torch.float32).to(device)

# aan_model = torch.jit.trace(model, dummy_data)
# aan_model = torch.jit.script(model)

# aan_model.save('aan.pt')
# loaded = torch.jit.load('aan.pt')

# print(loaded)

train.neurotreeFit(model, config, neuroloader_train, epochs=20)


# valid.neurotreeEvaluate(model, neuroloader_test, {'classification' : lambda x, y: 1 if x.argmax() == y else 0})

# valid.evaluate(model, neuroloader_test)


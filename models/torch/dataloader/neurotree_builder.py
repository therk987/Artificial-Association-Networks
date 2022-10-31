from representation.neurotree_builder import code, image, sound, text
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
import torch
from config.option import device

def neurotreeBuilder(batch_x, domains, subtasks, labels, data2neurotree):
    neurotrees = []
    node_pred = False
    if labels is not None:
        node_pred = True

    for i, (x, domain, task) in enumerate(zip(batch_x, domains, subtasks)):
        neurotree = data2neurotree[domain](x, domain)

        # if domain == 'image':
        #     if node_pred:
        #         neurotree = image.buildImageGT(x.to(device), task, labels[i])
        #     else:
        #         neurotree = image.buildImageGT(x, task)
        #
        #
        # elif domain == 'text':
        #     neurotree = text.buildTextGT(x, task, labels)
        # elif domain == 'class':
        #     neurotree = code.buildCodeGT(x, task)
        # elif domain == 'sound':
        #     neurotree = sound.buildImageGT(x, task)
        # else:
        #     assert EnvironmentError

        neurotrees.append(neurotree)

    return BatchNeuroTree(neurotrees)

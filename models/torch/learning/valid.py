from collections import defaultdict
from models.torch.networks.artificial_association_networks.artificial_association_networks import  ArtificialAssociationNeuralNetworks
from models.torch.dataloader.neurotree_builder import neurotreeBuilder
from models.torch.dataloader.batch_neurotree import BatchNeuroTree

import torch


def neurotreeEvaluate(model: ArtificialAssociationNeuralNetworks, dataloader, taskMetric):
    task_counts = defaultdict(int)
    task_corrects = defaultdict(int)
    model.train()
    for i, data in enumerate(dataloader):
        batch_x, domains, tasks, labels = data
        batchNeuroTree = neurotreeBuilder(batch_x, domains, tasks, labels)
        batchNeuroTree = model(batchNeuroTree, node_level=True)
        task_corrects, task_counts = recursiveMetric(batchNeuroTree, taskMetric)

    print(task_corrects, task_counts)
    for key in task_counts.keys():
        if key == 'classification':
            print(f"{key} : {task_corrects[key] / task_counts[key] * 100}% ")

    return task_corrects, task_counts


def nodeMetric(batchNeuroTree:BatchNeuroTree, taskMetric):
    task_results = defaultdict(int)
    task_counts = defaultdict(int)

    for i in range(len(batchNeuroTree.nodes)):
        task_name = batchNeuroTree.get_i('t_t', i)
        if task_name is not None:
            pred = batchNeuroTree.get_i('pred', i)
            label = batchNeuroTree.get_i('label', i)

            task_results[task_name] += taskMetric[task_name](pred, label)
            task_counts[task_name] += 1
    return task_results, task_counts

def recursiveMetric(batchNeuroTree:BatchNeuroTree, taskMetric):
    task_results, task_counts = nodeMetric(batchNeuroTree, taskMetric)
    child_count = batchNeuroTree.get_child_count()
    for i in range(child_count):
        child_neuro_tree, _ = batchNeuroTree.get_child(i)
        child_task_results, child_task_counts = recursiveMetric(child_neuro_tree, taskMetric)
        task_results = dict_sum(task_results, child_task_results)
        task_counts = dict_sum(task_counts, child_task_counts)
    return task_results, task_counts

def dict_sum(dict_1, dict_2):
    #  dict_1 << dict_2
    for key in dict_2.keys():
        dict_1[key] += dict_2[key]
    return dict_1



def evaluate(model:ArtificialAssociationNeuralNetworks, dataloader, node_level = False):
    task_counts = defaultdict(int)
    task_corrects = defaultdict(int)
    model.train()
    for i, data in enumerate(dataloader):
        batch_x, domains, tasks, labels = data
        batchNeuroTree = neurotreeBuilder(batch_x, domains, tasks, labels)
        h_root = model(batchNeuroTree, node_level = node_level)
        outputs = model.multi_task_networks.task_networks['classification'](batchNeuroTree)
        _, predictions = torch.max(outputs, 1)
        # 각 분류별로 올바른 예측 수를 모읍니다
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                task_corrects[label.item()] += 1
            task_counts[label.item()] += 1
    for i in range(len(task_counts.keys())):
        print(i, task_corrects[i]/task_counts[i] * 100)
    print(sum(task_corrects.values())/sum(task_counts.values()) * 100)





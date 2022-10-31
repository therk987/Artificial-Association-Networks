from collections import defaultdict
from models.torch.networks.artificial_association_networks import ArtificialAssociationNeuralNetworks
from models.torch.learning.config import TrainConfigure
from models.torch.dataloader.neurotree_builder import neurotreeBuilder
import torch

from config.option import device

import time

def multi_mode_switch(models, mode):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


def grouping_domain_items(data):
    batch_x, domains, subtasks, maintasks, labels = data
    group_items = defaultdict(lambda:defaultdict(list))
    for x, domain, subtask, maintask, label in zip(batch_x, domains, subtasks, maintasks, labels):
        group_items[domain]['x'].append(x)
        group_items[domain]['y'].append(label)

    return group_items

def neurotreeFit(models, domain_models, config:TrainConfigure, dataloader, testloader, epochs=5):

    test_counts = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    test_corrects = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))

    for epoch in range(epochs):
        task_counts = defaultdict(lambda:defaultdict(int))
        task_corrects = defaultdict(lambda:defaultdict(int))

        multi_mode_switch(models, 'train') # multi model.train()
        multi_mode_switch(domain_models.values(), 'train') # multi model.train()

        for i, data in enumerate(dataloader):
            print(i)
            batch_x, domains, subtasks, maintasks, labels = data

            config.optimizer.zero_grad()

            batchNeuroTree = neurotreeBuilder(batch_x, domains, subtasks, labels, data2neurotree=config.data2neurotree)
            domain_items = grouping_domain_items(data)

            for model in models:
                maintask_outputs, h_root, batchNeuroTree = model(batchNeuroTree, maintasks, node_level = True)
                sub_task_losses, sub_task_corrects, sub_task_counts = config.subtaskLoss(batchNeuroTree)
                main_task_losses, main_task_corrects, main_task_counts = config.maintaskLoss(maintasks, domains, maintask_outputs, labels)
                sub_loss = sum(sub_task_losses.values())
                main_loss = sum(main_task_losses.values())
                loss = sub_loss + main_loss
                loss.backward()
                config.optimizer.step()
                print(sub_task_corrects)
                print(main_task_corrects)
                print(main_task_counts)
                for task in main_task_counts.keys():
                    for domain in main_task_counts[task].keys():
                        print(f"{domain}:"
                              f"{main_task_corrects[task][domain]}/{main_task_counts[task][domain]},"
                              f" {main_task_corrects[task][domain]/main_task_counts[task][domain]}%")
                        task_counts[task][domain] += main_task_counts[task][domain]
                        task_corrects[task][domain] += main_task_corrects[task][domain]
                print(loss)

            for domain_name in domain_items.keys():
                domain_inputs = domain_items[domain_name]['x']
                domain_targets = domain_items[domain_name]['y']
                outputs = domain_models[domain_name](domain_inputs)
                loss = F.nll_loss(outputs, domain_targets)
                loss.backward()
                config.optimizer.step()
                print(sub_task_corrects)
                print(main_task_corrects)
                print(main_task_counts)
                for task in main_task_counts.keys():
                    for domain in main_task_counts[task].keys():
                        print(f"{domain}:"
                              f"{main_task_corrects[task][domain]}/{main_task_counts[task][domain]},"
                              f" {main_task_corrects[task][domain]/main_task_counts[task][domain]}%")
                        task_counts[task][domain] += main_task_counts[task][domain]
                        task_corrects[task][domain] += main_task_corrects[task][domain]
                print(loss)







        # params = model.state_dict()
        # torch.save(params, "models/pretrained/artificial-association-networks/torch-aan2.prm", pickle_protocol=4)
        # params = model.state_dict()
        # torch.save(params, "models/pretrained/모델명/모델파일.prm", pickle_protocol=4)

        # print(loss)
        print(f'[EPOCHS :{epoch + 1}]')
        for task in task_counts.keys():
            for domain in task_counts[task].keys():
                print(f"{domain}:"
                      f"{task_corrects[task][domain]}/{task_counts[task][domain]},"
                      f" {task_corrects[task][domain]/task_counts[task][domain]}%")
        test_corrects[epoch+1], test_counts[epoch+1] = neurotreeTest(model, testloader, config)
        print(f'[EPOCHS :{epoch + 1}]')
        # for i in range(len(task_counts.keys())):
        #     if task_counts[i] != 0:
        #         print(i, task_corrects[i]/task_counts[i] * 100)
        # task_count = sum(task_counts.values())
        # if task_count != 0:
        #     print(sum(task_corrects.values())/(task_count) * 100)
        # current9 = time.time()
        # config.scheduler.step()
    return test_corrects, test_counts


def neurotreeTest(model, dataloader, config):
    model.eval()
    task_counts = defaultdict(lambda: defaultdict(int))
    task_corrects = defaultdict(lambda: defaultdict(int))

    for i, data in enumerate(dataloader):
        # print(i)
        batch_x, domains, subtasks, maintasks, labels = data

        # current0 = time.time()
        batchNeuroTree = neurotreeBuilder(batch_x, domains, subtasks, labels, data2neurotree=config.data2neurotree)
        maintask_outputs, h_root, batchNeuroTree = model(batchNeuroTree, maintasks, node_level=True)
        # sub_task_losses, sub_task_corrects, sub_task_counts = config.subtaskLoss(batchNeuroTree)
        main_task_losses, main_task_corrects, main_task_counts = config.maintaskLoss(maintasks, domains,
                                                                                     maintask_outputs, labels)
        # print(sub_task_corrects)
        # print(main_task_corrects)
        # print(main_task_counts)
        # print('TEST:')
        for task in main_task_counts.keys():
            for domain in main_task_counts[task].keys():
                # print(f"{domain}:"
                #       f"{main_task_corrects[task][domain]}/{main_task_counts[task][domain]},"
                #       f" {main_task_corrects[task][domain] / main_task_counts[task][domain]}%")
                task_counts[task][domain] += main_task_counts[task][domain]
                task_corrects[task][domain] += main_task_corrects[task][domain]

    # params = model.state_dict()
    # torch.save(params, "models/pretrained/artificial-association-networks/torch-aan2.prm", pickle_protocol=4)
    # params = model.state_dict()
    # torch.save(params, "models/pretrained/모델명/모델파일.prm", pickle_protocol=4)

    # print(loss)
    print('[TEST]')
    for task in task_counts.keys():
        for domain in task_counts[task].keys():
            print(f"{domain}:"
                  f"{task_corrects[task][domain]}/{task_counts[task][domain]},"
                  f" {task_corrects[task][domain] / task_counts[task][domain]}%")
    print('[TEST]')
    return task_corrects, task_counts



import time
def multi_Fit(models, config:TrainConfigure, dataloader, testloader, epochs=5):
    for epoch in range(epochs):
        task_counts = defaultdict(lambda:defaultdict(int))
        task_corrects = defaultdict(lambda:defaultdict(int))
        # params = torch.load("models/pretrained/artificial-association-networks/torch-aan.prm", map_location=device)
        # model.load_state_dict(params)
        for model in models:
            model.train()

        for i, data in enumerate(dataloader):
            print(i)
            batch_x, domains, subtasks, maintasks, labels = data

            config.optimizer.zero_grad()

            # current0 = time.time()
            batchNeuroTree = neurotreeBuilder(batch_x, domains, subtasks, labels, data2neurotree=config.data2neurotree)
            # current1 = time.time()
            # print('time1', current1-current0)
            for model in models:

                maintask_outputs, h_root, batchNeuroTree = model(batchNeuroTree, maintasks, node_level = True)
                # current2 = time.time()
                # print('time2', current2-current1)
                sub_task_losses, sub_task_corrects, sub_task_counts = config.subtaskLoss(batchNeuroTree)
                # current3 = time.time()
                main_task_losses, main_task_corrects, main_task_counts = config.maintaskLoss(maintasks, domains, maintask_outputs, labels)
                # print('time3', current3-current2)
                sub_loss = sum(sub_task_losses.values())
                main_loss = sum(main_task_losses.values())
                loss = sub_loss + main_loss
                # current4 = time.time()
                # print('time4', current4-current3)
                loss.backward()
                # current5 = time.time()
                # print('time5', current5-current4)
                config.optimizer.step()
                # current6 = time.time()
                # print('time6', current6-current5)


                # current8 = time.time()
                print(sub_task_corrects)
                print(main_task_corrects)
                print(main_task_counts)
                for task in main_task_counts.keys():
                    for domain in main_task_counts[task].keys():
                        print(f"{domain}:"
                              f"{main_task_corrects[task][domain]}/{main_task_counts[task][domain]},"
                              f" {main_task_corrects[task][domain]/main_task_counts[task][domain]}%")
                        task_counts[task][domain] += main_task_counts[task][domain]
                        task_corrects[task][domain] += main_task_corrects[task][domain]
                print(loss)
        for task in task_counts.keys():
            for domain in task_counts[task].keys():
                print(f"{domain}:"
                      f"{task_corrects[task][domain]}/{task_counts[task][domain]},"
                      f" {task_corrects[task][domain]/task_counts[task][domain]}%")

        neurotreeTest(model, testloader, config)



def multiTest(models, dataloader, config):
    for model in models:
        model.eval()

    task_counts = defaultdict(lambda: defaultdict(int))
    task_corrects = defaultdict(lambda: defaultdict(int))

    for i, data in enumerate(dataloader):
        # print(i)
        batch_x, domains, subtasks, maintasks, labels = data

        # current0 = time.time()
        batchNeuroTree = neurotreeBuilder(batch_x, domains, subtasks, labels, data2neurotree=config.data2neurotree)
        maintask_outputs, h_root, batchNeuroTree = model(batchNeuroTree, maintasks, node_level=True)
        # sub_task_losses, sub_task_corrects, sub_task_counts = config.subtaskLoss(batchNeuroTree)
        main_task_losses, main_task_corrects, main_task_counts = config.maintaskLoss(maintasks, domains,
                                                                                     maintask_outputs, labels)

        for task in main_task_counts.keys():
            for domain in main_task_counts[task].keys():
                task_counts[task][domain] += main_task_counts[task][domain]
                task_corrects[task][domain] += main_task_corrects[task][domain]


    print('[TEST]')
    for task in task_counts.keys():
        for domain in task_counts[task].keys():
            print(f"{domain}:"
                  f"{task_corrects[task][domain]}/{task_counts[task][domain]},"
                  f" {task_corrects[task][domain] / task_counts[task][domain]}%")
    print('[TEST]')




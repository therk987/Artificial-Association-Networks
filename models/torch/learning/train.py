from collections import defaultdict
from models.torch.networks.artificial_association_networks.artificial_association_networks import ArtificialAssociationNeuralNetworks
from models.torch.learning.config import TrainConfigure
from models.torch.dataloader.neurotree_builder import neurotreeBuilder
import torch
from config.option import device

def fit(model:ArtificialAssociationNeuralNetworks, config:TrainConfigure, dataloader, node_level = False, epochs=5):

    for epoch in range(epochs):
        task_counts = defaultdict(int)
        task_corrects = defaultdict(int)
        model.train()
        for i, data in enumerate(dataloader):
            batch_x, domains, tasks, labels = data
            config.optimizer.zero_grad()
            batchNeuroTree = neurotreeBuilder(batch_x, domains, tasks, labels, config.data2neurotree)
            # current0 = time.time()
            h_root = model(batchNeuroTree, node_level = node_level)
            # current1 = time.time()
            outputs = model.multi_task_networks.task_networks['classification'](batchNeuroTree)
            # current2 = time.time()
            loss = config.loss(outputs, labels)
            # current3 = time.time()
            loss.backward()
            # current4 = time.time()
            config.optimizer.step()
            # current5 = time.time()

            # current6 = time.time()
            _, predictions = torch.max(outputs, 1)
            # 각 분류별로 올바른 예측 수를 모읍니다
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    task_corrects[label.item()] += 1
                task_counts[label.item()] += 1
            # current7 = time.time()

            # print('time1', current1-current0)
            # print('time2', current2-current1)
            # print('time3', current3-current2)
            # print('time4', current4-current3)

        # current8 = time.time()
        for i in range(len(task_counts.keys())):
            print(i, task_corrects[i]/task_counts[i] * 100)
        print(sum(task_corrects.values())/sum(task_counts.values()) * 100)
        # current9 = time.time()
        # config.scheduler.step()

import time
def neurotreeFit(model:ArtificialAssociationNeuralNetworks, config:TrainConfigure, dataloader, testloader, epochs=5):

    test_counts = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    test_corrects = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))

    for epoch in range(epochs):
        task_counts = defaultdict(lambda:defaultdict(int))
        task_corrects = defaultdict(lambda:defaultdict(int))
        # params = torch.load("models/pretrained/artificial-association-networks/torch-aan.prm", map_location=device)
        # model.load_state_dict(params)
        model.train()

        for i, data in enumerate(dataloader):
            print(i)
            batch_x, domains, subtasks, maintasks, labels = data

            config.optimizer.zero_grad()

            # current0 = time.time()
            batchNeuroTree = neurotreeBuilder(batch_x, domains, subtasks, labels, data2neurotree=config.data2neurotree)
            # current1 = time.time()
            # print('time1', current1-current0)
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

        # params = model.state_dict()
        # torch.save(params, "models/pretrained/artificial-association-networks/torch-aan2.prm", pickle_protocol=4)
        # params = model.state_dict()
        # torch.save(params, "models/pretrained/모델명/모델파일.prm", pickle_protocol=4)

        # print(loss)
        for task in task_counts.keys():
            for domain in task_counts[task].keys():
                print(f"{domain}:"
                      f"{task_corrects[task][domain]}/{task_counts[task][domain]},"
                      f" {task_corrects[task][domain]/task_counts[task][domain]}%")

        neurotreeTest(model, testloader, config)
        # for i in range(len(task_counts.keys())):
        #     if task_counts[i] != 0:
        #         print(i, task_corrects[i]/task_counts[i] * 100)
        # task_count = sum(task_counts.values())
        # if task_count != 0:
        #     print(sum(task_corrects.values())/(task_count) * 100)
        # current9 = time.time()
        # config.scheduler.step()


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



#
#
# def train(training_loader, valid_loader, test_loader, epochs=100):
#     # Train
#     global tttt
#
#     networks = [ran_model]
#     task_networks = [ran_task_model]
#     optimizers = [ran_optimizer]
#     schedulers = [ran_scheduler]
#     model_names = ['ran_', 'raan_']
#     for epoch in range(epochs):
#         ran_model.train()
#         ran_task_model.train()
#         is_training = True
#
#         running_loss = 0.0
#         correct = 0.0
#         ctn = 0
#         task_counts = defaultdict(int)
#         task_corrects = defaultdict(int)
#         for i, data in enumerate(training_loader):
#             # get the inputs
#             tree = buildDatasets(*data)
#             correct_sum = 0
#             networks_i = 0
#             optimizers[networks_i].zero_grad()
#
#             h_root, treeset = ran_model.propagation(tree)
#             tttt = treeset
#             #             print()
#             result = ran_task_model.working([treeset[-1]])
#             #             print(result)
#             losses = ran_task_model.loss(result)
#             print(losses)
#             loss = losses[2]
#             loss.backward()
#
#             optimizers[networks_i].step()
#             running_loss += loss.item()
#             print(loss.item(), running_loss, correct_sum)
#
#             correct += correct_sum
#             ctn += len(tree.nodes)
#             if i % 100 == 0:
#                 print('\Train set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
#                     correct, ctn,
#                     100. * correct / ctn), running_loss)
#                 task_percentage = [(_, task_corrects[_] / task_counts[_]) for _ in task_counts.keys()]
#                 for key in task_counts.keys():
#                     task_cnt = task_counts[key]
#                     if task_cnt == 0:
#                         task_cnt = 1
#                     print(key, str(task_corrects[key]) + '/' + str(task_counts[key]),
#                           task_corrects[key] / task_cnt * 100)
#         train_task_counts.append(task_counts)
#         train_task_corrects.append(task_corrects)
#         for scheduler in schedulers:
#             scheduler.step()
#         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(running_loss))
#         #         test(test_loader)
#
#         running_loss = 0.0
#     return 100. * correct / ctn
#     # test(lineGraphCode, trainloader, device)
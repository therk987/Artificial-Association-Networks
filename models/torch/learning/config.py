import torch
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR

from config.option import device
torchOptimizer = {
    'adam' : torch.optim.Adam,
}




# torchLoss = {
#     'crossentropy': torch.nn.CrossEntropyLoss(),
#     'mse': torch.nn.MSELoss()
# }

# torchMetrix = {
#
# }
# torchLossWeights = {
#
# }
# torchRunEagerly = {
#
# }




class TrainConfigure():

    def __init__(self,
                 model,
                 multi_subtask,
                 multi_maintask,
                 optimizer='adam',
                 lr=0.01,

                 # loss_weights=None,
                 # metrics=None,
                 # weighted_metrics=None,
                 # run_eagerly=None,
                 # steps_per_execution=None,
                 ):

        global torchOptimizer
        global torchLoss
        global torchMetrix
        global torchLossWeights
        global torchRunEagerly

        self.parameters = model.parameters()
        self.lr = lr

        self.optimizer = torchOptimizer[optimizer](self.parameters,
                                                   lr=self.lr,
                                                   # weight_decay = 1e-05
                                                   )


        self.multi_subtask = multi_subtask
        self.multi_maintask = multi_maintask

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=2, eta_min=1e-05)

        # self.loss = {}

        # if type(sub_task_loss) == dict:
        #     for key, value in sub_task_loss.items():
        #         self.loss[key] = torchLoss[value]
        # else:
        #     assert Exception



    def node_loss(self, batchNeuroTree:BatchNeuroTree):
        task_losses = defaultdict(int)
        task_corrects = defaultdict(lambda:defaultdict(int))
        task_counts = defaultdict(lambda:defaultdict(int))

        batch_neurotree = defaultdict(list)
        batch_domain = defaultdict(list)
        batch_output = defaultdict(list)
        batch_targets = defaultdict(list)

        for i in range(len(batchNeuroTree.nodes)):
            task_name = batchNeuroTree.get_i('t_t', i)
            if task_name != None:

                batch_domain[task_name].append(batchNeuroTree.nodes[i].t_d)
                batch_neurotree[task_name].append(batchNeuroTree.nodes[i])
                # print('task!!', batchNeuroTree.nodes[i].label)
                batch_targets[task_name].append(torch.Tensor([batchNeuroTree.nodes[i].label])[0])
                batch_output[task_name].append(batchNeuroTree.nodes[i].pred.squeeze(0))

            # if task_name != None or task_name != 'layer':
            #     task_output = batchNeuroTree.get_i('pred', i)
            #     task_label = batchNeuroTree.get_i('label', i)
            #     batch_task_outputs[task_name].append(task_output)
            #     batch_task_labels[task_name].append(task_label)

        for task_name, task_batch_neurotree in batch_neurotree.items():
            # if type(batch_targets[task_name]) == list:
            #     task_batch_targets = torch.Tensor(batch_targets[task_name]).to(device, dtype = torch.long)
            # else:
            #     task_batch_targets = torch.stack(batch_targets[task_name]).to(device, dtype = torch.long)
            # task_batch_targets = torch.stack(batch_targets[task_name]).to(device, dtype = torch.long)

            # task_losses[task_name], task_corrects[task_name] = self.multi_subtask.task_networks[task_name].loss(
            #     BatchNeuroTree(task_batch_neurotree),
            #     task_batch_targets
            # )
            task_losses[task_name], task_corrects[task_name], task_counts[task_name] = self.multi_subtask.loss(
                task_name,
                batch_domain[task_name],
                batch_output[task_name],
                batch_targets[task_name]
            )

            #
            #
            #
            # print(task_corrects)
            # print(task_counts)
            # print(len(task_batch_neurotree))
            # task_counts[task_name] += len(task_batch_neurotree)

        return task_losses, task_corrects, task_counts

    def recursive_loss_for_subtask(self, allBatchNeuroTree:BatchNeuroTree):

        batchNeuroTree = allBatchNeuroTree.get_nodes_to_calculate_sub_loss()
        task_losses, task_corrects, task_counts = self.node_loss(batchNeuroTree)
        if len(batchNeuroTree.nodes) == 0 :
            return task_losses, task_corrects, task_counts

        child_count = batchNeuroTree.get_child_count()
        for i in range(child_count):
            child_neuro_tree, _ = batchNeuroTree.get_child(i)
            child_task_losses, child_task_corrects, child_task_counts = self.recursive_loss_for_subtask(child_neuro_tree)
            #             loss += child_loss
            task_losses = self.dict_sum(task_losses, child_task_losses)
            # print('correct',task_corrects)
            # print('count',task_counts)
            task_corrects = self.double_dict_sum(task_corrects, child_task_corrects)
            task_counts = self.double_dict_sum(task_counts, child_task_counts)
        return task_losses, task_corrects, task_counts

    def subtask_loss(self, allBatchNeuroTree:BatchNeuroTree):
        return self.recursive_loss_for_subtask(allBatchNeuroTree)

    def maintask_loss(self, tasks, domains, maintask_outputs, targets):
        task_losses = defaultdict(int)
        task_corrects = defaultdict(int)
        task_counts = defaultdict(int)

        batch_domain = defaultdict(list)
        batch_output = defaultdict(list)
        batch_targets = defaultdict(list)

        for i, task_name in enumerate(tasks):
            if task_name != None:
                batch_domain[task_name].append(domains[i])
                batch_output[task_name].append(maintask_outputs[i])
                batch_targets[task_name].append(targets[i])

        for task_name, task_batch_outputs in batch_output.items():
            # if task_name == 'classification':
            # print(batch_task_outputs[task_name].shape, batch_task_labels[task_name].shape)
            task_losses[task_name], task_corrects[task_name], task_counts[task_name] = self.multi_maintask.loss(
                task_name,
                batch_domain[task_name],
                task_batch_outputs,
                batch_targets[task_name]
            )
            #
            # elif task_name == 'autoencoder':
            #     batchNeuroTree.init_loss_count()
            #     task_losses[task_name] = self.autoencoderLoss(batchNeuroTree)


        return task_losses, task_corrects, task_counts


    def autoencoderLoss(self, allBatchNeuroTree:BatchNeuroTree):
        loss = 0
        batchNeuroTree = allBatchNeuroTree.get_nodes_to_calculate_main_loss()
        if len(batchNeuroTree.nodes) == 0 :
            return loss

        domain_list = batchNeuroTree.get('t_d')
        x_list = batchNeuroTree.get('x')
        pred_list = batchNeuroTree.get('pred')
        for domain, x, pred in zip(domain_list, x_list, pred_list):
            if domain != None:
                # print(domain, x, pred)
                loss += self.loss['autoencoder'](x, pred)
        child_count = batchNeuroTree.get_child_count()
        for i in range(child_count):
            child_neuro_tree, _ = batchNeuroTree.get_child(i)
            loss += self.autoencoderLoss(child_neuro_tree)
        return loss

    def dict_sum(self, dict_1, dict_2):
        #  dict_1 << dict_2
        for key in dict_2.keys():
            dict_1[key] += dict_2[key]
        return dict_1

    def double_dict_sum(self, dict_1, dict_2):
        #  dict_1 << dict_2
        for key1 in dict_2.keys():
            for key2 in dict_2[key1].keys():
                dict_1[key1][key2] += dict_2[key1][key2]
        return dict_1
from torch import nn
import torch
from collections import defaultdict
from config.option import device

class Classification(torch.nn.Module):
    def __init__(self, hidden_dim, class_count):
        super().__init__()
        self.task_name = 'classification'
        self.hidden_dim = hidden_dim
        self.class_count = class_count

        self.loss_function = torch.nn.NLLLoss()


        self.recognition_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, class_count),
            nn.LogSoftmax(dim=-1)
            #                     nn.Softmax(dim = 1)
        )

    def forward(self, hiddens, tree):
        if type(hiddens) == list:
            hiddens = torch.stack(hiddens, dim = 0)
        outputs = self.recognition_layer(hiddens)
        return outputs

    def loss(self,batch_domains, batch_outputs, batch_targets):
        # corrects = 0
        preds = torch.stack(batch_outputs, dim = 0)
        targets = torch.stack(batch_targets, dim=0).to(device, dtype = torch.long)

        # print('loss',preds.shape, targets.shape)

        corrects = defaultdict(int)
        counts = defaultdict(int)


        loss = self.loss_function(preds, targets)
        _, predictions = torch.max(preds, 1)

        for domain, label, prediction in zip(batch_domains, targets, predictions):
            if label == prediction:
                corrects[domain] += 1
            counts[domain] += 1
        return loss, corrects, counts
    #
    # def loss(self,batch_domains, batch_outputs, batch_targets):
    #     # version 2
    #     # corrects = 0
    #     preds = torch.stack(batch_outputs, dim = 0)
    #     targets = torch.stack(batch_targets, dim=0).to(device, dtype = torch.long)
    #
    #     # print('loss',preds.shape, targets.shape)
    #
    #     losses = defaultdict(int)
    #     corrects = defaultdict(int)
    #     counts = defaultdict(int)
    #
    #
    #     # loss = self.loss_function(preds, targets)
    #     _, predictions = torch.max(preds, 1)
    #
    #     for idx, (domain, label, prediction) in enumerate(zip(batch_domains, targets, predictions)):
    #         losses[domain] += self.loss_function(preds[idx].unsqueeze(0), targets[idx].unsqueeze(0))
    #         if label == prediction:
    #             corrects[domain] += 1
    #         counts[domain] += 1
    #     return losses, corrects, counts

    # def loss_i(self, tree, i):
    #     # print(tree.h[i].shape)
    #
    #     pred = self.recognition_layer(tree.h[i].unsqueeze(0))
    #     y = tree.y[i].to(device, dtype=torch.long)
    #     correct = self.correct(pred, y)
    #     # print(pred.shape, correct)
    #     return F.nll_loss(pred, y), correct
    #
    # def loss(self, tree):
    #     # pred, y = items
    #     pred = tf.stack(tree.get('pred'), axis=0)
    #     y = tf.concat(tree.get('y'), axis=0)
    #
    #     correct = self.correct(pred, y)
    #     # print(pred.shape, y.shape)
    #     return F.nll_loss(pred, y), correct
    #
    # def correct(self, pred, y):
    #     task_pred = pred.max(1, keepdim=True)[1]
    #     task_correct_count = task_pred.eq(y.view_as(task_pred)).sum().item()
    #     return task_correct_count

import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

def add_weight_decay(model, weight_decay=1e-5, lr=0.01):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0., 'lr': lr},
        {'params': decay, 'weight_decay': weight_decay, 'lr': lr}]

class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        parameters = []
        base_lr = optim_dict['learning_rate'].pop('default')
        for n, p in model.named_children():
            lr_ = base_lr
            for m, lr in optim_dict['learning_rate'].items():
                if m in n:
                    lr_ = lr
            parameters.append({'params':p.parameters(), 'lr':lr_})
            print(n, lr_)
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['optim_args']['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
                **self.optim_dict['optim_args']
            )
        elif self.optim_dict["optimizer"] == 'AdamW':
            self.optimizer = optim.AdamW(
                parameters,
                **self.optim_dict['optim_args']
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer)

    def define_lr_scheduler(self, optimizer):
        if self.optim_dict["optimizer"] in ['SGD', 'Adam', 'AdamW']:
            if self.optim_dict['scheduler_type'] == 'MultiStepLR':
                basic_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, 
                    **self.optim_dict['scheduler_args'],
                    )
            elif self.optim_dict['scheduler_type'] == 'CosineAnnealingLR':
                basic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    **self.optim_dict['scheduler_args'],
                    )
            warm_up_epoch = self.optim_dict.pop('warm_up_epoch', 0)
            if warm_up_epoch > 0:
                lambda1 = lambda epoch: 0.8 ** (warm_up_epoch - epoch)
                wr_scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda1, last_epoch=-1,
                    )
                lr_scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, 
                    schedulers=[wr_scheduler, basic_scheduler], 
                    milestones=[warm_up_epoch]
                    )
            else:
                lr_scheduler = basic_scheduler
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
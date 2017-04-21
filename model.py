# Taken from https://github.com/ikostrikov/pytorch-a3c
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class ES(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs):
        """
        Really I should be using inheritance for the small_net here
        """
        super(ES, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.train()

    def forward(self, inputs):

        x = F.elu(self.linear1(inputs))
        x = F.elu(self.linear2(x))
        x = F.elu(self.linear3(x))

        return self.actor_linear(x)

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]

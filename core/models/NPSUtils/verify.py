from EncoderDecoder import MNISTModel
from utilities.GroupLinearLayer import GroupLinearLayer
from BabyAIModels import VanillaModel
from ModifiedModels import RuleNetwork
from torch import nn

import torch
import argparse
import time

parser = argparse.ArgumentParser()


class m(nn.Module):

    def __init__(self):

        super(m, self).__init__()

        self.linear = nn.Linear(3, 10, bias=False)

        # recurrent part to maintain a hidden state of slots
        self.rnn = nn.ModuleList([nn.GRU(10, 10, batch_first=True) for _ in range(7)])
        self.prev_h = None

    def forward(self, x):

        bs, w, h, d = x.shape

        if self.prev_h is None:
            self.prev_h = [torch.zeros(1, bs, 10) for _ in range(len(self.rnn))]

        return x

a = m()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# a = VanillaModel()
print(count_parameters(a.linear), len(a.rnn))


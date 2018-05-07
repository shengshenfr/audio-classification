import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

EPOCH = 1

INPUT_SIZE = 160
HIDDEN_SIZE = 60
N_LAYERS = 2
N_CLASSES = 2




class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers = N_LAYERS,
            batch_first=True,
        )

        self.out = nn.Linear(HIDDEN_SIZE, N_CLASSES)
    def forward(self, x):

        r_out, (h_n, h_c) = self.rnn(x, None)


        out = self.out(r_out[:, -1, :])
        return out

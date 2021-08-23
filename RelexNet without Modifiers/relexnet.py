import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from data_loader import DataLoader
from collections import Counter, defaultdict

from torch.autograd import Variable


class RelexNet(nn.Module):
    # Single step RNN.
    # input_size is char_vacab_size=26,hidden_size is number of hidden neurons，output_size is number of categories
    def __init__(self, vocabulary_size, len_seq, num_classes):
        super(RelexNet, self).__init__()
        self.num_classes = num_classes
        self.L = nn.Linear(vocabulary_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, B_layer):
        # X.shape = (batch, seq_len, vocab_size)
        T = input.shape[1]
        batch = input.shape[0]
        predict_y = Variable(torch.zeros(batch, self.num_classes))

        if B_layer is None:
            B_layer = Variable(torch.zeros(batch, self.num_classes)).cuda()

        for t in range(T):
            tmp = input[:, t, :]

            L_onestep = self.L(tmp)
            L_onestep = self.dropout(L_onestep)
            # L_onestep = torch.sigmoid(L_onestep)
            L_onestep = F.relu6(L_onestep)

            B_layer = torch.add(B_layer, L_onestep)
            # print(B_layer)

            if self.num_classes == 1:
                predict_y[t] = F.sigmoid(B_layer)
            else:
                predict_y = self.softmax(B_layer)

        return predict_y, B_layer

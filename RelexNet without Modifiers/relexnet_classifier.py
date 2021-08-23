import os
import torch.optim

from relexnet import RelexNet
# from models import FastText, LSTM
from training import train_model
from sklearn.utils import shuffle
import os
import glob
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

num_epochs = 20
batch_size = 32  # Number of hidden neurons in model
context_window = 1
k = 9

dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # If you have a GPU installed, use that, otherwise CPU
print(dev)
print('Loading data...')
all_files = glob.glob("data/sentiment/*.csv")
# all_files = glob.glob("*.csv")
data = pd.concat((pd.read_csv(f, header=None, index_col=None) for f in all_files))
# path = os.path.join('data', 'sentiment', 'amazon_cells_labelled.csv')
# data = pd.read_csv(path)
data.columns = ['sentence', 'label']
data = data.sample(frac=1, random_state=1)
# data = shuffle(data)
dataset = DataLoader(data, k=k, batch_size = batch_size, context_window=context_window, dev=dev)
# print(f'Data Size: {dataset.df.size()}')
print("Data Ready!")
vocabulary_size = len(dataset.vocab)
len_seq = dataset.max_length
num_classes = 2

# model = FastText(len(dataset.token_to_id)+2, num_hidden, len(dataset.class_to_id)).to(dev)
model = RelexNet(vocabulary_size=vocabulary_size, len_seq=len_seq, num_classes=num_classes).cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.9,0.99))
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.9,0.99), weight_decay=10)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[8 * dataset.no_batch, 14 * dataset.no_batch],gamma = 0.1, last_epoch=-1)

losses, accuracies = train_model(dataset, model, optimizer, scheduler, num_epochs, dev=dev)
print(losses)
# torch.save(model, os.path.join('classifier.pth'))
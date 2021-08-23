import numpy as np
import pandas as pd
import torch
from preprocessor import Preprocessor
from collections import Counter, defaultdict
from sklearn.preprocessing import OneHotEncoder
import torch.utils.data as Data
from modifiers import Modifiers


class DataLoader:
    def __init__(self, data, batch_size, k, context_window=1, preprocessor=Preprocessor(), enc_tokens=OneHotEncoder(),
                 dev='cpu'):
        self.df = data
        self.dev = dev
        self.batch_size = batch_size
        self.context_window = context_window
        # self.modifiers = modifier.BOOSTER_DICT.keys()
        # self.enc_modifiers = enc_modifiers
        # self.vocab_frequency = 5
        self.padding_length = 75
        self.fold_num = 10
        # self.padding_length = 192
        # self.first_sentence = 128
        # self.second_sentence = self.padding_length - self.first_sentence

        self.apply_preprocessor(preprocessor)
        enc_tokens.fit(self.vocab)
        # self.split(0.4, 0.3)
        # print(len(self.test))
        self.split(0.9, 0.1)

        # self.k_fold_partition(fold_num = self.fold_num) # Divide ten folds
        # print("K Fold Partition Completed!")
        # train, validation = self.k_fold_split(k)

        train_num = len(self.train)
        if train_num % self.batch_size == 0:
            self.no_batch = train_num / self.batch_size
        else:
            self.no_batch = int(train_num / self.batch_size) + 1

        trainX, trainY = self.build_training_data(enc_tokens, self.train)
        # trainX, trainY = self.build_training_data(enc_tokens, train)

        # validationX, validationY = self.build_training_data(enc_tokens, validation)
        testX, testY = self.build_training_data(enc_tokens, self.test)
        print(f'Train Dataset Shape : {trainX.shape}')
        # print(f'Validation Dataset Shape : {validationX.shape}')
        print(f'Test Dataset Shape : {testX.shape}')
        self.train_dataset = self.get_batch(trainX, trainY)
        # self.validation_dataset = self.get_batch(validationX, validationY)
        self.test_dataset = self.get_batch(testX, testY)

    def get_batch(self, X, Y):
        dataset = Data.TensorDataset(X, Y)
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=2,
        )
        return loader
        # train_dataset = Data.TensorDataset(self.train_X, self.train_Y)
        # valid_dataset = Data.TensorDataset(self.valid_X, self.valid_Y)
        # test_dataset = Data.TensorDataset(self.test_X, self.test_Y)
        # self.loader = Data.DataLoader(
        #     dataset=train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     # num_workers=2,
        # )
        # self.valid_dataset = Data.DataLoader(
        #     dataset=valid_dataset,
        #     # batch_size=self.batch_size,
        #     shuffle=True,
        #     # num_workers=2,
        # )
        # self.test_dataset = Data.DataLoader(
        #     dataset=test_dataset,
        #     # batch_size=self.batch_size,
        #     shuffle=True,
        #     # num_workers=2,
        # )

    def remove_low_frequency(self, list):
        new_list = []
        for x in list:
            if x in self.token_to_count.keys():
                new_list.append(x)
        return new_list

    def apply_preprocessor(self, preprocessor):
        self.df['tokens'] = [preprocessor(s) for s in self.df['sentence']]
        self.df['tokens'] = [x[:self.padding_length] if len(x) > self.padding_length else x for x in self.df['tokens']]

        # for index, row in self.df.iterrows():
        #     if len(row['tokens']) > self.max_length:
        #         row[toke]

        self.token_to_count = Counter([x for l in self.df['tokens'] for x in l])

        # tmp_token_to_count = self.token_to_count.copy()
        # for index, value in tmp_token_to_count.items():
        #     if value <= self.vocab_frequency:
        #         self.token_to_count.pop(index)
        # self.df['tokens'] = [self.remove_low_frequency(x) for x in self.df['tokens']]
        self.max_length = self.get_max_length()
        print(f'Max Length : {self.max_length}')

        self.vocab = list([[term] for term in self.token_to_count.keys()])

        print(f'Vocab Size : {len(self.vocab)}')

    def get_max_length(self):
        max_length = 0
        for index, row in self.df.iterrows():
            ######## Changed part for negatives ###########
            token_list = [x for x in row['tokens']]
            ##############      End       #################
            tmp_length = len(token_list)
            if tmp_length > max_length:
                max_length = tmp_length
        return max_length

    def k_fold_partition(self, fold_num=10):
        # index = list(range(len(self.df))) # total index
        batch_size = int(len(self.train) / fold_num)  # the number of data for each fold
        remain_num = len(self.train) - batch_size * fold_num  # the remain data after partition
        self.fold_data = []
        fold_batch_list = []  # Average the remaining data to the folds
        for fold in range(fold_num):
            if remain_num > 0:
                remain_num -= 1
                fold_batch_list.append(batch_size + 1)
            else:
                fold_batch_list.append(batch_size)
        fold_index = 0  # The starting position of each division data
        for fold in range(fold_num):
            fold_texts = [fold_index, fold_index + fold_batch_list[fold]]
            self.fold_data.append(fold_texts)
            fold_index = fold_index + fold_batch_list[fold]

    def k_fold_split(self, k):
        print(f'K : {k}, fold_data[k] : {self.fold_data[k]}')
        validation = self.train.iloc[self.fold_data[k][0]: self.fold_data[k][1]]
        train = self.train.iloc[0: self.fold_data[k][0]].append(self.train.iloc[self.fold_data[k][1]:])
        train_num = len(self.train) - (self.fold_data[k][1] - self.fold_data[k][0])
        if train_num % self.batch_size == 0:
            self.no_batch = train_num / self.batch_size
        else:
            self.no_batch = int(train_num / self.batch_size) + 1
        print(f' index : {self.no_batch}')
        return train, validation

    def split(self, train, test):
        index = int(train * len(self.df))
        self.train = self.df.iloc[0:index]
        self.test = self.df.iloc[index:]
        # if self.train_index % self.batch_size == 0:
        #     self.no_batch = self.train_index / self.batch_size
        # else:
        #     self.no_batch = int(self.train_index / self.batch_size) + 1
        # print(f' index : {self.no_batch}')

    def build_training_data(self, enc, df):
        X = []
        Y = []
        for index, row in df.iterrows():
            tmp = [enc.transform([[t]]).toarray()[0] for t in row['tokens']]
            if len(tmp) < self.max_length:
                pad_length = self.max_length - len(tmp)
                for i in range(pad_length):
                    tmp.append(np.zeros(len(self.vocab)))
            X.append(tmp)
            Y.append(row['label'])
        X = np.array(X)
        Y = np.array(Y)
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        print(f'Input Data Shape (sequence_num, sequence_len, vocab_size + 1) : {X.shape}')
        print(f'Input Label Shape : {Y.shape}')
        return X, Y





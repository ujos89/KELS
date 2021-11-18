import torch
import numpy as np
import os
import pandas as pd

import torch.utils.data as D

np.random.seed(42)

def get_year(series):
    nan_index = series.index[series.notna()].tolist()
    year = sorted(list(set([int(_[3]) for _ in nan_index])))
    year_col = ['L2Y'+str(y) for y in year]
    
    return year, year_col
    

class KELS(D.Dataset):
    def __init__(self, root_dir='./preprocessed/merge/outer'):
        self.root_dir = root_dir
        self.input_df = pd.read_csv(os.path.join(self.root_dir, 'input_merge.csv')).set_index('L2SID')
        self.label_df = pd.read_csv(os.path.join(self.root_dir, 'label_merge.csv')).set_index('L2SID')
        
    def __len__(self):
        if len(self.label_df) == len(self.input_df):
            return len(self.label_df)
    
        return False
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_series = self.input_df.iloc[idx]
        label_series = self.label_df.iloc[idx]
        year, year_col = get_year(label_series)
        
        input, label = {}, {}
        label6_fit = {1:4.0, 2:4.0, 3:3.0, 4:3.0, 5:3.0, 6:2.0, 7:2.0, 8:1.0, 9:1.0}

        for y, y_col in zip(year, year_col):
            label_index = [ _ for _ in label_series.index if _.startswith(y_col)]
            input_index = [ _ for _ in input_series.index if _.startswith(y_col)]

            input_y = input_series[input_index].to_dict()
            label_y = label_series[label_index].to_dict()

            if y == 6:
                label_y = {k:label6_fit[int(label_y[k])] for k in label_y}

            input[y], label[y] = input_y, label_y

        sample = {'year':year, 'input':input, 'label':label}
        return sample

class SplitDataset(D.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = torch.Tensor(indices)

    def __len__(self):
        return self.indices.size(0)
    
    def __getitem__(self, item):
        idx = self.indices[item]
        return self.dataset[idx]
        
class SSGDSampler(D.Sampler):
    r"""Samples elements according to SSGD Sampler
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, model, batch_size):
        self.training_data = data_source.train_data.to(device)
        self.training_label = data_source.train_labels.to(device)
        self.training_data = self.training_data.view(self.training_data.shape[0], 1, self.training_data.shape[1], self.training_data.shape[2])
        self.training_data = self.training_data.type(torch.cuda.FloatTensor)
        self.model = model
        self.batch_size = batch_size


    def compute_score(self):
        sampled=[]
        print(self.training_data.shape)
        output = model(self.training_data)
        loss = F.cross_entropy(output, self.training_label, reduce=False)
        prob = F.softmax(loss)
        feat = model.feat
        for _ in range(0, self.batch_size):
            if len(sampled)==0:
                 sampled.extend(torch.argmax(prob))
            else:
                dist = torch.mm(self.feat, self.feat[sampled].T)
                min_dist = torch.min(dist, dim=0)
                mean_dist = torch.mean(dist, dim=0)
                score = min_dist + mean_dist + prob
                max_idx = torch.argmax(score)
                sampled.extend(max_idx)
        return sampled

    def __iter__(self):
        sampled=self.compute_score()
        print(sampled)
        return iter(sampled)

    def __len__(self):
        return len(self.data_source)

        

def train_val_test_split(dataset, test_size=500, val_ratio=.2):
    # KELS dataset size : 7156
    # first split test set with size (less than 872, # of complete data)
    # then, split train validation set with ratio
    # return : split dataset
    
    test_size = min(872, test_size) 
    complete_indices, incomplete_indices = np.array([]), np.array([])
    
    for idx, sample in enumerate(dataset):
        if sample['year'] == [1,2,3,4,5,6]:
            complete_indices = np.append(complete_indices, idx)
        else:
            incomplete_indices = np.append(incomplete_indices, idx)
    
    test_indices = np.random.choice(complete_indices, test_size, replace=False)    
    train_val_indices = np.concatenate([np.setdiff1d(complete_indices, test_indices), incomplete_indices])
    
    val_size = int(len(train_val_indices)*val_ratio)
    val_indices = np.random.choice(train_val_indices, val_size, replace=False)
    train_indices = np.setdiff1d(train_val_indices, val_indices)
        
    return SplitDataset(dataset, train_indices), SplitDataset(dataset, val_indices), SplitDataset(dataset, test_indices)
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from utils.nn_utils import make_splited_data, KELSDataSet
from transformer import train_net

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(233,80),
            nn.ReLU(),
            nn.Linear(80,30),
            nn.ReLU(),
            nn.Linear(30,9)
        )
    def forward(self,x):
        x[torch.isnan(x)] = 0
        return self.mlp(x)


MLP = MLP()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_datapaths = ['./preprocessed/prepared/nan/L2Y1.pkl','./preprocessed/prepared/nan/L2Y2.pkl','./preprocessed/prepared/nan/L2Y3.pkl','./preprocessed/prepared/nan/L2Y4.pkl','./preprocessed/prepared/nan/L2Y5.pkl','./preprocessed/prepared/nan/L2Y6.pkl',]
label_datapath = './preprocessed/prepared/nan/label.pkl'

# read pickle
input_datas = [] # list of each input pandas dataframe
for datapath in X_datapaths:
    temp = pd.read_pickle(datapath)
    temp = temp.reset_index()
    input_datas.append(temp)

label_data = pd.read_pickle(label_datapath)
label_data = label_data.reset_index()
seq_len = len(input_datas)

label_data = label_data - 1
CLS2IDX = {
    0 : '1등급',
    1 : '2등급',
    2 : '3등급',
    3 : '4등급',
    4 : '5등급',
    5 : '6등급',
    6 : '7등급',
    7 : '8등급',
    8 : '9등급'
}
is_regression = False

# build dataset
X_trains, X_tests, y_train, y_test = make_splited_data(input_datas,label_data,is_regression=is_regression)
train_dataset = KELSDataSet(X_trains,y_train,is_regression=is_regression)
test_dataset = KELSDataSet(X_tests,y_test,is_regression=is_regression)

batch_size = 32
hidden_features = 100
embbed_dim = 72

#build dataloader
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


epochs = 40

train_net(MLP,train_loader,test_loader,n_iter=epochs,device=device,mode='E',lr=0.0001)
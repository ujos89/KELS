import torch
import torch.utils.data as D
import pretty_errors
import argparse
from rich import print
from algorithms.RNN import *
from utils.dataloader import *

# bulid dataset
root_dir = './preprocessed/merge/outer'
dataset = KELS(root_dir=root_dir)
train_sampler, val_sampler, test_sampler = train_val_test_split(dataset, test_size=300, val_ratio=.2)

train_loader = D.DataLoader(dataset=dataset, sampler=train_sampler, shuffle=False)
val_loader = D.DataLoader(dataset=dataset, sampler=val_sampler, shuffle=False)
test_loader = D.DataLoader(dataset=dataset, sampler=test_sampler, shuffle=False)

# print(len(train_loader))  -> 5485
# print(len(val_loader))    -> 1371
# print(len(test_loader))   -> 300

# parser = argparse.ArgumentParser()
# parser.add_argument('--cpu', action='store_true',help='run in cpu') 
# args = parser.parse_args()
# GPU_NUM = 0
# if args.cpu:
#     device = torch.device('cpu')
# else:
#     device = torch.device('cuda')
#     print(torch.cuda.get_device_name(GPU_NUM), "allocated in ", torch.cuda.current_device())

# hyperparameters for training
epochs = 1

# model = VanillaRNN(input_size=input_size,
#                    hidden_size=hidden_size,
#                    sequence_length=sequence_length,
#                    num_layers=num_layers,
#                    device=device).to(device)

def sample2tensor(sample):
    if not sample['year']:
        print("="*10,'YEAR NAN DETECTED', "="*10)
        return
    
    year = torch.cat(sample['year'])
    input = sample['input']
    
    for y in year:
        input_ = torch.cat([v for k, v in input[int(y)].items()])
        
        if torch.any(torch.isnan(input_)):
            print("="*10,'INPUT NAN DETECTED', "="*10)
            continue
        
        print(y, input_)
        
    ###### nan exception tratment #### 
        

for epoch in range(1, epochs+1):
    for idx, sample in enumerate(train_loader):
        sample2tensor(sample)   

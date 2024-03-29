import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
import pretty_errors
import argparse
from rich import print
from algorithms.MLP import *
from utils.dataloader import *


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true',help='run in cpu') 

GPU_NUM = 0
args = parser.parse_args()
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(GPU_NUM), "allocated in ", torch.cuda.current_device())
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'

# bulid dataset
# root_dir = './preprocessed/merge/outer'
root_dir = './preprocessed/merge/inner'
# root_dir = '../KELS_data/preprocessed/merge/outer'

dataset = KELS(root_dir=root_dir)
# train_sampler, val_sampler, test_sampler = train_val_test_split(dataset, test_size=300, val_ratio=.2)   #outer
train_sampler, val_sampler, test_sampler = train_val_test_split(dataset, test_size=50, val_ratio=.2)  #inner

train_loader = D.DataLoader(dataset=dataset, sampler=train_sampler, shuffle=False)
val_loader = D.DataLoader(dataset=dataset, sampler=val_sampler, shuffle=False)
test_loader = D.DataLoader(dataset=dataset, sampler=test_sampler, shuffle=False)

### INNER
# print(len(train_loader))    -> 568
# print(len(val_loader))      -> 142
# print(len(test_loader))     -> 50

### OUTER
# print(len(train_loader))  -> 5485
# print(len(val_loader))    -> 1371
# print(len(test_loader))   -> 300

def sample2tensor(sample, standard=25):
    if not sample['year']:
        # print("="*20,'YEAR NAN DETECTED', "="*20)
        return
    
    year = torch.cat(sample['year'])
    input = sample['input']
    label = sample['label']
    
    flag = False
    # years, inputs, labels = [], [], []
    
    for y in year:
        input_ = torch.cat([v for k, v in input[int(y)].items()])
        # order of label (E, K, M)
        label_ = torch.cat([v for k, v in label[int(y)].items()])
        
        if torch.any(torch.isnan(input_)) or torch.any(torch.isnan(label_)):
            print("="*10,'INPUT NAN DETECTED', "="*10)
            print(sample)
    
        else:
            if not flag:
                flag = True
                years, inputs, labels = y.unsqueeze(0), input_[:standard].float(), label_.float()
            else:
                years = torch.cat([years, y.unsqueeze(0)])
                inputs = torch.cat([inputs, input_[:standard]])
                labels = torch.cat([labels, label_])
                
                
            # years.append(int(y))
            # inputs.append(input_)
            # labels.append(label_)

    return years, inputs.resize_(years.shape[0], standard), labels.resize_(years.shape[0], 3)
        
# hyperparameters for training
epochs = 5
batch_size = 1
input_size = 150
hidden_size = 30
target_size = 4

model = KELS_MLP(input_size=input_size, hidden_size=hidden_size, target_size=target_size, batch_size=batch_size, device=device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=.1)
criterion = nn.CrossEntropyLoss()

#save parameters
path_save = './models/MLP'
save_period = 10
best_acc = 0
best_model = model

for epoch in range(1, epochs+1):
    # TRAIN
    model.train()
    for idx, sample in enumerate(train_loader):
        samples = sample2tensor(sample)
        if not samples is None:
            year, input, label = samples
            model.zero_grad()
            
            # label for English
            label = label[:, 0].unsqueeze(-1)-1
            label = F.one_hot(label.to(torch.int64), num_classes=4).squeeze(1).to(device)
            label = torch.tensor(label, dtype=torch.float32).to(device)
            label = label[-1] #label for last year

            output = model(input)            
            loss = criterion(output.unsqueeze(0), label.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        
    # print log
    print("EPOCH: %d / %d, LOSS: %f" % (epoch, epochs, loss.item()))
            
    if epoch % save_period == 0:
        torch.save(model, os.path.join(path_save,'MLP'+str(epoch)+'.pt'))
        
    # VALIDATION
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            samples = sample2tensor(sample)
            if not samples is None:
                year, input, label = samples
                
                # get label (English grade)
                label = label[:, 0].unsqueeze(-1)-1
                label = F.one_hot(label.to(torch.int64), num_classes=4).squeeze(1).to(device)
                label = torch.tensor(label, dtype=torch.float32).to(device)
                label = label[-1]
                
                output = model(input)
                
                pred = torch.argmax(output)
                label = torch.argmax(label)
                
                # predict last year
                if pred == label:
                    correct += 1
                total += 1 

    acc = correct / total
    print("EPOCH: %d / %d, ACCURACY: %.6f (%d / %d)" % (epoch, epochs, acc, correct, total))
        
    if acc > best_acc:
        best_acc = acc
        best_model = model
        
# torch.save(best_model, os.path.join(path_save,'MLP_best.pt'))
        
#TEST
total, correct = 0, 0
with torch.no_grad():
    for idx, sample in enumerate(test_loader):
        samples = sample2tensor(sample)
        if not samples is None:
            year, input, label = samples
            
            # get label (English grade)
            label = label[:, 0].unsqueeze(-1)-1
            label = F.one_hot(label.to(torch.int64), num_classes=4).squeeze(1).to(device)
            label = torch.tensor(label, dtype=torch.float32).to(device)
            label = label[-1]
            
            output = best_model(input)
            pred = torch.argmax(output)
            label = torch.argmax(label)
            
            if pred == label:
                correct += 1
            total += 1

acc = correct / total            
print("EPOCH: %d / %d, ACCURACY: %.6f (%d / %d)" % (epoch, epochs, acc, correct, total))

            

# import the best trained model
# testing the model with test data

# GT data  < > output test results
# test Acc.
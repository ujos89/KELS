import torch
import torch.utils.data as D

from utils.dataloader import *

# bulid dataset
root_dir = './preprocessed/merge/outer'
dataset = KELS(root_dir=root_dir)
print(dataset[3])
# train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, test_size=300, val_ratio=.2)


idx=348
print(dataset[idx])
print(len(dataset[idx]['input'][1]))
print(len(dataset[idx]['input'][2]))
print(len(dataset[idx]['input'][3]))
print(len(dataset[idx]['input'][4]))
print(len(dataset[idx]['input'][5]))
print(len(dataset[idx]['input'][6]))

exit()
###### DEBUG : TypeError: Cannot index by location index with a non-integer key
###### TRY bulid custom sampler to split

# hyperparameters for dataloader
batch_size = 4

# train_loader = D.DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
# val_loader = D.DataLoader(dataset=dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)
# test_loader = D.DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

# # hyperparameters for training
# epochs = 5

# for epoch in range(1, epochs+1):
#     for idx, sample in enumerate(train_loader):
#         print(sample)
#         break
    
#     break

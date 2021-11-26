import torch
import torch.nn as nn
import torch.nn.functional as F

class KELS_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(KELS_LSTM, self).__init__()
        
        # batch size is 1 (sequence length is variable)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.layer_num = 1
        
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, batch_first=True)
        
        
    def forward(self, input):
        input = torch.tensor(input, dtype=torch.float32).to(self.device)
        input = input.unsqueeze(0)
        h0 = torch.zeros(self.layer_num, self.batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.layer_num, self.batch_size, self.hidden_size).to(self.device)
        
        output, _ = self.lstm(input, (h0, c0))
        output = F.softmax(output[0], dim=1)
        output = torch.argmax(output, dim=1, keepdim=True)
        
        ##ADD FC LAYER TO LABEL DETECT
        
        return output
    

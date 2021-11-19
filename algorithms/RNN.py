## RNN moduel with first 5 year
import torch
import torch.nn as nn



inputs_test = torch.Tensor(1, 10, 5)
print(inputs_test.shape[2])

# one way
# exit()
def oneRNN(input, hidden_size) : 
  input_size = (input.shape[2]) # input shape
  # (batch_size, time_steps, input_size)
  cell = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = 2, batch_first=True)
  outputs, _status = cell(input)
  return outputs

# bi direc.
def biRNN(input, hidden_size) : 
  input_size = (input.shape[2]) # input shape
  # (batch_size, time_steps, input_size)
  cell = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = 2, batch_first=True, bidirectional = True)
  outputs, _status = cell(input)
  return outputs


output = oneRNN(inputs_test,8)


print(output.shape)



import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

T = 1000
time = torch.arange(1, T + 1, dtype = torch.float32) # 1到1000为时间
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
print(x)
d2l.plot(time, [x], 'time', 'x', xlim=[1,1000], figsize=(6,3))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    
def get_net():
    net = nn.Sequential(nn.Linear(4,10),nn.ReLU(),nn.Linear(10,1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss()

F.one_hot()

nn.RNN()
nn.GRU()
nn.LSTM()

nn.Embedding

a = "adv"
a.lower()

nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
optimizer.zero_grad()
optimizer.step()

nn.LayerNorm(2)
nn.BatchNorm1d(2)
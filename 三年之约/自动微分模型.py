import torch
from torch import torch_version

x=torch.ones(2,5)
print(f'x:{x}')
y=torch.zeros(2,3)
w=torch.randn(5,3,requires_grad=True)
b=torch.randn(3,requires_grad=True)
z=x@w+b
criterion=torch.nn.MSELoss()
loss=criterion(z,y)
loss.sum().backward()
print(f'w的梯度：{w.grad}')
print(f'b的梯度：{b.grad}')

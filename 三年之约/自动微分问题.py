"""
detach()
问题：张量设置了自动微分，就不能直接转成NP数组的ndarray对想了

"""
import torch
import numpy as np
t1=torch.tensor(10,requires_grad=True,dtype=torch.float)
print(f't1:{t1},type:{type(t1)}')
t2=t1.detach()
print(f't2:{t2},type:{type(t2)}')
t1.data[0]=100
print(f't1:{t1},type:{type(t1)}')
print(f't2:{t2},type:{type(t2)}')
print(f't1:{t1.requires_grad}')
print(f't2:{t2.requires_grad}')
n1=t2.numpy()
print(f'n1:{n1},type:{type(n1)}')
#******n2=t1.detach().numpt()链式编程直接解决*********





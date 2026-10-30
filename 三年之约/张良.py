"""
装量
torch.tensor
torch.Tensor
torch.Inttensor
"""
#import torch
#import numpy as np
#def wjt666():
#尝试直接创建张良
"""
创建线性和随机张良
torch.arange()和torch.linspace()创建线性设置
torch.random.initial_seed和torch.random.manual_seed()随机种子设置
torch.rand/randn()创建随机浮点数类型张良
torch.randint(low,high,size())创建随机整数类型张良
"""
import  torch
def dm01():
    t1=torch.arange(1,11,2)#惨3为步长
    print(f't1={t1},type=type{t1}')
    t2=torch.linspace(1,10,5)#惨3为元素个数
    print(f't2={t2},type=type{t2}')
#随机
def dm02():
    #随机种子
    #torch.initial_seed()
    torch.manual_seed(3)#设计随机种子
    #均匀分布的随即张良
    t1=torch.rand(size=(2,3))
    print(f't1={t1},type=type{t1}')
    print('_'*30)
    #符合正态分布的随机张良
    t2=torch.randn(size=(2,3))
    print(f't2={t2},type=type{t2}')
    print('_'*30)
    #创建随机整数张良
    t3=torch.randint(0,10,size=(3,5))
    print(f't3={t3},type=type{t3}')
    print('_' * 30)
"""
创建全0，1的张良
torch.ones和torch.ones_like 创建全1张良
torch.zeros和torch。zeros_like 创建全0张良*******************
torch.full和torch.full_like 创建全为指定值张良
"""
t4=torch.ones(2,3)
print(f't4={t4},type=type{t4}')
print('_' * 30)
t5=torch.tensor([[1,2], [3,4], [5,6]])
print(f't5={t5},type=type{t5}')
print('_' * 30)
t6=torch.ones_like(t5)
print(f't6={t6},type=type{t6}')
print('_' * 30)
t4=torch.zero_(2,3)
print(f't4={t4},type=type{t4}')
print('_' * 30)
t5=torch.tensor([[1,2], [3,4], [5,6]])
print(f't5={t5},type=type{t5}')
print('_' * 30)
t6=torch.zeros_like(t5)
print(f't6={t6},type=type{t6}')
print('_' * 30)
t4=torch.full((2,3),255)
print(f't4={t4},type=type{t4}')
print('_' * 30)
t5=torch.tensor([[1,2], [3,4], [5,6]])
print(f't5={t5},type=type{t5}')
print('_' * 30)
t6=torch.full_like(t5,255)
print(f't6={t6},type=type{t6}')
print('_' * 30)






if __name__=='__main__':
        dm02()

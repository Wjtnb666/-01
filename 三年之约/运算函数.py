"""
sum()
max()
min()
mean()
都有dim参数，0表示列，1表示行
pow(),sqrt(),exp(),log(),log2(),log10() 没有dim参数
"""
import torch
t1=torch.tensor ([[1,2,3],[2,3,4]],dtype=torch.float)
print(f't1={t1}')
print(t1.sum(dim=1))
print(t1.sum(dim=0))
print(t1.sum())
print(t1.max(dim=1))
print(t1.max(dim=0))
print(t1.max())
print(t1.min(dim=1))
print(t1.min(dim=0))
print(t1.min())
print(t1.mean(dim=1))
print(t1.mean(dim=0))
print(t1.mean())
print(t1.pow(2))
print(t1**2)#效果通上
print(t1.sqrt())
print(t1.exp())#e的n次幂
print(t1.log())
print(t1.log10())

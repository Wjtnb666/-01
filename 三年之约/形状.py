"""
reshape()不改变内容
squeze()    维度
unsqueze()  删除维度为一的维度
transpose()一次交换2个维度
permute()一次交换多个维度
view()
contiguous()
is_contiguous()
"""
from idlelib import __main__

import torch

def wjt():
    t1=torch.randint(1,10,size=(2,3))
    print(f't1={t1},shape={t1.shape},row={t1.shape[0]},lie={t1.shape[1],t1.shape[-1]}')
    t2=t1.reshape(1,6)
    print(f't2={t2},shape={t2.shape}')
#    t2=t1.reshape(2,4)#报错数据太多
def wjt1():
    t1=torch.randint(1,10,size=(2,3))
    print(f't1={t1},shape={t1.shape},row={t1.shape[0]},lie={t1.shape[1],t1.shape[-1]}')
    t2=t1.unsqueeze(0)
    print(f't2={t2},shape={t2.shape}')
    t3 = t1.unsqueeze(1)
    print(f't3={t3},shape={t3.shape}')
    t4= t1.unsqueeze(2)
    print(f't4={t4},shape={t4.shape}')
    t5 = torch.randint(1,10,size=(2,1,3,2,1,1))
    print(f't5={t5},shape={t5.shape}')
    t6=t5.squeeze()
    print(f't5={t5},shape={t5.shape}')
def wjt2():
    t1 = torch.randint(1, 10, size=(2, 3,4))
    print(f't1={t1},shape={t1.shape}')
    t2=t1.transpose(0,1)
    print(f't2={t2},shape={t2.shape}')
    print(f't1={t1}')
    t3=t1.permute(2,0,1)
    print(f't3={t3},shape={t3.shape}')











 if __name__=='__main__':
     #wjt()
     #wjt1()
     wjt2()
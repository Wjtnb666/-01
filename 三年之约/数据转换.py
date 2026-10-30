"""
创建指定类型张良
type(torch)
half()/double()/float()/short()/int()/long()
"""
import torch
from torch.ao.quantization import float16_static_qconfig

#创建指定类型
t1=torch.tensor([1,2,3,4,5,6],dtype=torch.float)
print(f't1={t1},type={t1.dtype}')
t2=t1.type(torch.int16)
print(f't2={t2},type={t2.dtype}')
#************************
print(t2.half())#float16
print(t2.float())#float32
print(t2.double())#float64
print(t2.short())#int16
print(t2.int())#32
print(t2.long())#64



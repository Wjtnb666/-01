"""
张良专np
*张良对象.numpy()  共想
张良对象.numpy().copy()  bu共享
2.np,nd数组转张良
from_nump() 共想
*torch.tensor(nd数组)  bu共享
3.从标量张良 提取起内容
*标量张良。itme()
"""
import torch
import numpy as np
def dm01():
    # 1.创建张良
    t1=torch.tensor([1,2,3,4,5])
    print(f't1={t1},type={type(t1)}')
    # 2. 张量转np
    n1=t1.numpy()
    #n1=t1.numpy().copy() 不共享
    print(f't1={n1},type={type(n1)}')
    # 3. 共享内存
    n1[0]=100
    print(f'n1={n1}')
    print(f't1={t1}')
    n2=np.array([11,22,33])
    print(f'n2={n2},type={type(n2)}')
    t1=torch.from_numpy(n2)
    t2=t1.type(torch.float32)
    print(f't1:{t1},type={type(t1)}')
    print(f't2:{t2},type={type(t2)}')
    t1=torch.tensor(100)
    value=t1.item()
    print(f'value={value},type={type(value)}')


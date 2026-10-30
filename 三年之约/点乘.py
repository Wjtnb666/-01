import torch
#点乘 mul
#t1=torch.tensor([[1,2,3],[3,4,5],[5,6,7]])
#print(f'{t1}')
#t2=torch.tensor([[3,3,3],[2,2,2],[4,4,4]])
#print(f'{t2}')
#t3=t2*t1
#print(f'{t3}')
#矩阵相乘 matmul
#A行==B列
#2行3列
t1=torch.tensor([[1,2,3],[3,4,5]])
#3行2列
t2=torch.tensor([[3,3], [2,2], [7,7]])
t3=t1 @ t2
print(f'{t3}')





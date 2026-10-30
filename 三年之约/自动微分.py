"""
权重更新公式
w新=w旧-学习率*梯度
"""
import torch
#只有标量张量才能求导，默认float
#参1=初始值，参2=是否自动微分，参3=数据类型
w=torch.tensor(10,requires_grad=True,dtype=torch.float)
#定义损失函数
loss=2*w**2
loss.sum().backward()#计算梯度后，会记录到w.grad属性中
print(f'梯度函数类型={type(loss.grad_fn)}')
#带入权重更新公式
w.data=w.data-0.01*w.grad
print(f'更新后的权重={w}')







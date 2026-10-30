"""


add() sub() mul() div() neg()
add_() sub_() mul_() div_() neg_()加减乘除，取反****修改元数据





"""
import torch
t1=torch.tensor([1,2,3])
#t2=t1.add(10)***不修改原数据
#t2=t1+10***同上
#t1.add_(10)#***修改原数据
#t1+=10#效果同上
t2=t1.sub(1)#减法
t2=t1.mul(2)#乘法
t2=t1.div(2)#除法
t2=t1.neg()#取反3变成-3
print(f'{t1},{t2}')


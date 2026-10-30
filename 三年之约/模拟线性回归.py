import torch
from sympy.tensor import tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
def creat_dataset():
    x,y,coef=make_regression(
        n_samples=100,
        n_features=1,
        noise=10,
        coef=True,
        bias=14.5,
        random_state=3
    )
    x=torch.tensor(x,dtype=torch.float)
    y=torch.tensor(y,dtype=torch.float)
    return x,y,coef
def train(x,y,coef):
    #创建数据级对象
    dataset=TensorDataset(x, y)
    #创建数据加载对象
    #参1=数据集对象，参2=批次大小，参3=是否打乱数据
    dataloader=DataLoader(dataset,batch_size=16,shuffle=True)
    #创建初始的线形回归模型
    #参一=特征 参2=输出特征
    model=nn.Linear(in_features=1,out_features=1)
    #创建损失函数
    criterion=nn.MSELoss()
    #创建优化器
    #参1=模型参数，参2=学习率
    optimizer=optim.SGD(model.parameters(),lr=0.01)
    #训练过程
    #定义变量 训练次数，每轮的损失值，训练总损失，训练的样本数
    epochs,loss_list,total_loss,total_sample=100,[],0.0,0
    for epoch in range (epochs):
        for train_x,train_y in dataloader:
            y_pred= model(train_x)
            #计算每批的平均损失值
            loss=criterion(y_pred,train_y.reshape(-1,1))#-1自动计算
            total_loss+=loss.item()
            total_sample+=1
            #梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'轮数={epochs}，平均损失{total_loss/total_sample}')
        loss_list.append(total_loss / total_sample)
    print(f'{epochs}平均损失{loss_list}')
    print(f'权重{model.weight},偏执{model.bias}')
    #绘制
    plt.plot(range(epochs),loss_list)
    plt.title('损失值曲线变化图')
    plt.grid()#绘制网格线
    plt.show()
    #绘制样本点分布
    plt.scatter(x,y)
    y_pred=torch.tensor(data=[v*model.weight+model.bias for v in x])
    y_true=torch.tensor(data=[v*coef+14.5 for v in x])
    plt.plot(x,y_pred,color='red',label='预测值')
    plt.plot(x,y_true,color='green',label='真实值')
    plt.legend()#图例
    plt.show()

    #print(type(x))
if __name__=='__main__':
    #创建数据集
    x,y,coef=creat_dataset()
   # print(f'x:{x},y:{y},coef:{coef}')
    #模型训练
    train(x, y, coef)
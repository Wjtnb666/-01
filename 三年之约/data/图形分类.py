import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

from 三年之约.自动微分模型 import criterion

BATCH_SIZE = 64
#准备数据集
def create_dataset():
    #参1=数据集保存路径，参2=是否是训练集，参3=数据预处理变成张量，参4=是否下载数据集
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    #   测试集
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset
class imgModule(nn.Module):
    #初始化父类
    def __init__(self):
        super().__init__()
        #搭建卷积层
        self.conv1=nn.Conv2d(3,6,3,1,0)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,3,1,0)
        self.pool2=nn.MaxPool2d(2,2)
        #全连接层
        self.linear1=nn.Linear(576,120)
        self.linear2=nn.Linear(120,84)
        self.linear3=nn.Linear(84,10)
    def forward(self,x):
        #卷积层1：加权求和+激活函数+降维
        x=self.conv1(x)
        x=torch.relu(x)
        x=self.pool1(x)
        #第二层
        x=self.conv2(x)
        x=torch.relu(x)
        x=self.pool2(x)
        #全连接，数据转化（8，16，6，6）->（8，576）
        #参1样本数（行数）参2：特征数（列数）参3：自动计算
        x=x.reshape(x.size(0),-1)#-1表示自动计算
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=self.linear3(x)
        return x
def train(model, train_dataset):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epoch=200
    for i in range(epoch):
        dataloader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)#shuffle=True打乱数据顺序
        sam_num=0#样本数量
        total_loss=0#总损失
        start=time.time()#开始时间
        #遍历数据训练
        for x,y in dataloader:
            # x：一个批次的图片，形状 (8, 3, 32, 32)，float32，数值 0~1
            # y：对应的标签，形状 (8,)，如 tensor([3, 9, 1, 0, ...])，取值 0~9
            optimizer.zero_grad()
            # PyTorch 默认会把每次 backward 算出的梯度"累加"到参数的 .grad 里。
            # 所以每批训练前必须清零，否则这批的梯度会和上一批混在一起。
            output=model(x)#前向传播
            loss=criterion(output,y)#损失
            loss.backward()#反向传播
            optimizer.step()#参数优化
            sam_num+=len(x)#样本数量
            total_loss+=loss.item()#总损失
        end=time.time()
        print(f'epoch:{i+1}, loss:{total_loss/sam_num}, time:{end-start}')
        #保存模型
        torch.save(model.state_dict(), 'model.pth')
#模型测试
def evaluate(model, test_dataset):
    #数据加载
    dataloader=DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model=imgModule()#模型
    model.load_state_dict(torch.load('model.pth'))#加载模型
    total_correct=0.0
    total_sample=0.0
    total_loss=0
    start=time.time()
    #正确率
    for x,y in dataloader:
        model.eval()#
        y_pred=model(x)
        y_pred=torch.argmax(y_pred, dim=-1)
        total_correct+=(y_pred==y).sum()
        total_sample+=len(y)
    acc=total_correct/total_sample
    print(f'test acc:{acc}')
    end=time.time()
    print(f'test loss:{total_loss/total_sample}, time:{end-start}')






if __name__ == '__main__':
   #搭建神经网络
    net = imgModule()
    train_dataset, test_dataset = create_dataset()
    #summary(net, (3, 32, 32),batch_size=BATCH_SIZE)
    #模型训练
    train(net, train_dataset)
   #模型测试
    evaluate(net, test_dataset)




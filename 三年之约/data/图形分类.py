import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
from tiny_imagenet_dataset import TinyImageNetTrain, TinyImageNetVal, val_transform

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#准备数据集
def create_dataset():
    train_full = TinyImageNetTrain()                        # 10万张，带增强
    val_full = TinyImageNetTrain(transform=val_transform)   # 同一批图，但不带增强
    val_size = int(len(train_full) * 0.1)                   # 1万张
    train_size = len(train_full) - val_size                 # 9万张
    generator1 = torch.Generator().manual_seed(42)
    generator2 = torch.Generator().manual_seed(42)
    train_set, _ = random_split(train_full, [train_size, val_size], generator=generator1)
    _, val_set = random_split(val_full, [train_size, val_size], generator=generator2)
    return train_set, val_set
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
        self.linear1=nn.Linear(3136,120)
        self.linear2=nn.Linear(120,84)
        self.linear3=nn.Linear(84,200)
    def forward(self,x):
        #卷积层1：加权求和+激活函数+降维
        x=self.conv1(x)
        x=torch.relu(x)
        x=self.pool1(x)
        #第二层
        x=self.conv2(x)
        x=torch.relu(x)
        x=self.pool2(x)
        #全连接，数据转化（8，16，14，14）->（8，3136）
        #参1样本数（行数）参2：特征数（列数）参3：自动计算
        x=x.reshape(x.size(0),-1)#-1表示自动计算
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=self.linear3(x)
        return x
def validate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    model.train()
    return correct / total

class imgModuleV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)  # 新增
            self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.linear1 = nn.Linear(16384, 120)  # 32→16→8
            self.linear2 = nn.Linear(120, 200)

        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = x.reshape(x.size(0), -1)
            x = self.dropout(torch.relu(self.linear1(x)))
            return self.linear2(x)
def train(model, train_dataset, val_dataset, epoch=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    best_acc = -1.0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    for i in range(epoch):
        sam_num=0#样本数量
        total_loss=0#总损失
        correct=0#本轮猜对的张数
        start=time.time()#开始时间
        #遍历数据训练
        for batch_idx,(x,y) in enumerate(train_loader):
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad()
            output=model(x)#前向传播
            loss=criterion(output,y)#损失
            loss.backward()#反向传播
            optimizer.step()#参数优化
            sam_num+=len(x)#样本数量
            total_loss+=loss.item()#总损失（每个batch的平均loss）
            correct+=(output.argmax(dim=1)==y).sum().item()

        end=time.time()
        val_acc = validate(model, val_loader)

        print(f'epoch:{i+1}, loss:{total_loss/len(train_loader):.4f}, '
              f'train acc:{correct/sam_num:.4f}, val_acc:{val_acc:.4f}, '
              f'lr:{optimizer.param_groups[0]["lr"]:.5f}, time:{end-start:.0f}s')

        torch.save(model.state_dict(), 'last.pth')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best.pth')
            print(f'新val_acc={val_acc:.4f}，已保存 best.pth')
        scheduler.step()   # 每个 epoch 末衰减一次，必须在 batch 循环外
    print(f'训练结束val_acc={best_acc:.4f}')


#模型测试
def evaluate(model, test_dataset):
    #数据加载
    dataloader=DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    model.load_state_dict(torch.load('best.pth', map_location=device))
    model.eval()
    criterion=nn.CrossEntropyLoss()
    total_correct=0.0
    total_sample=0.0
    total_loss=0.0
    start=time.time()
    with torch.no_grad():
        for x,y in dataloader:
            x,y=x.to(device),y.to(device)
            y_pred=model(x)
            total_loss+=criterion(y_pred,y).item()*len(x)# 乘批量，最后按样本数平均
            correct=(y_pred.argmax(dim=1)==y).sum().item()
            total_correct+=correct
            total_sample+=len(y)
    acc=total_correct/total_sample
    print(f'test acc:{acc:.4f} ({acc*100:.2f}%)')
    end=time.time()
    print(f'test loss:{total_loss/total_sample:.4f}, time:{end-start:.0f}s')





if __name__ == '__main__':
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    print(f'使用设备: {device}')
    net = imgModuleV2().to(device)
    train_set, val_set = create_dataset()
    print(net)
    train(net, train_set, val_set, epoch=40)
    evaluate(net, TinyImageNetVal())





"""
卷积神经网络
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import features


def dm01():
    img=plt.imread('帅比.JPG')
    print(f'img{img},shape{img.shape}')#(2316, 3088, 3)
    img2=torch.tensor(img,dtype=torch.float)
    img2=img2.permute(2,0,1)
    print(f'img2{img2},shape{img2.shape}')
    #保存图片，增加维度
    img3=img2.unsqueeze(dim=0)
    print(f'img3{img3},shape{img3.shape}')
    #创建卷积层参数1输入通道数，参数2输出通道数（几个腾征途），参数3卷积核大小，参数4步长，参数5填充
    conv=nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,stride=1,padding=1)
    conv_img=conv(img3)
    print(f'conv_img{conv_img},shape{conv_img.shape}')
    #可视化1通道
    img4=conv_img[0,:,:,:]
    print(f'img4{img4},shape{img4.shape}')
    img5=img4.permute(1,2,0)
    feature1=img5[:,:,1].detach().numpy()
    plt.imshow(feature1)
    plt.show()




if __name__ == '__main__':
    dm01()


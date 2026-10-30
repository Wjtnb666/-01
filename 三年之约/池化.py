"""
池化层不会
"""
import torch
import torch.nn as nn
def dm01():
    inputs=torch.tensor(
        [[
            [0,1,2],
            [4,5,6],
            [8,9,10],
            ],
            [
                [5, 16, 2],
                [40, 54, 6],
                [8, 97, 10],
            ],
            [
            [8,13,22],
            [47,5,63],
            [89,90,10],
            ]
        ])
    print(f'inputs={inputs},shape={inputs.shape}')
    #创建最大池化层
    pool1=nn.MaxPool2d(2,1,0)
    output=pool1(inputs)
    print(f'output={output},shape={output.shape}')
    #创建平均池化层
    pool2=nn.AvgPool2d(2,1,0)
    output=pool2(inputs)
    print(f'output={output},shape={output.shape}')




if __name__ == '__main__':
    dm01()




import torch
w=torch.tensor(10,requires_grad=True,dtype=torch.float)

for i in range (1,101):
    loss = w ** 2 + 20
    if w.grad is not None:
        w.grad.zero_()
    loss.sum().backward()
    w.data=w.data-0.01*w.grad
print(f'最终结果 权重：{w},梯度：{w.grad},loss:{loss}')

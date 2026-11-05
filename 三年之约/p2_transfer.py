"""
P2 迁移学习：resnet18 微调 Tiny ImageNet-200
================================================
用法：直接运行，不用改任何东西。
      python p2_transfer.py

与 V2/V4 的唯一区别：模型换成 resnet18（别人在 ImageNet 上训好的），
只训练最后一层 fc，前面所有层冻结不动。

预计：5 轮，10-20 分钟，准确率 55-65%。
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '4')

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

from tiny_imagenet_dataset import TinyImageNetTrain, TinyImageNetVal

# ---------- 超参数 ----------
BATCH_SIZE = 64      # 224 的图比 64 的图占显存，128 会爆
EPOCH = 5            # 只跑 5 轮，目的是看到数字，不是刷分
LR = 1e-3            # 比从零训练的 0.01 小 10 倍：参数已经很好，步子要小

SAVE_PATH = 'best_transfer.pth'   # 注意：不叫 best.pth，避免覆盖 V4 的存档


# ---------- 关键一：预处理必须和预训练时一致 ----------
# resnet18 在 ImageNet 上训练时：图缩到 224，用 ImageNet 的均值方差归一化。
# 你原来的 64x64 + Tiny-ImageNet 自己的均值方差（0.4802...）会跟权重对不上，
# 精度会莫名其妙掉几个点，而且不报错。
train_tf = transforms.Compose([
    transforms.Resize(224),                       # 64 -> 224，放大 3.5 倍
    transforms.RandomCrop(224, padding=16),       # 先补边再裁，等于随机平移
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def create_dataset():
    """和你原来的划分完全一致，保证对比公平（同一个 seed 42）。"""
    train_full = TinyImageNetTrain(transform=train_tf)
    val_full = TinyImageNetTrain(transform=val_tf)
    val_size = int(len(train_full) * 0.1)
    train_size = len(train_full) - val_size

    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    train_set, _ = random_split(train_full, [train_size, val_size], generator=g1)
    _, val_set = random_split(val_full, [train_size, val_size], generator=g2)
    return train_set, val_set


def build_model():
    torch.manual_seed(42)   # 固定随机性，方便复现

    # 关键二：weights=... 就是"把别人训好的参数下载下来"
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 换掉最后一层：原来是 Linear(512, 1000)，现在要 200 类
    model.fc = nn.Linear(512, 200)

    # 关键三：先全部冻结，只解冻 fc
    # requires_grad=False 表示"这层的参数不许改"
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set, val_set = create_dataset()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = build_model().to(device)

    # 打印参数账，这是要养成的习惯
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数      : {total:,}  ({total / 1e6:.2f}M)')
    print(f'可训练参数  : {trainable:,}  ({trainable / 1e6:.3f}M)')
    print(f'冻结比例    : {100 * (1 - trainable / total):.2f}%')
    print(f'训练集 {len(train_set)} | 验证集 {len(val_set)} | 设备 {device}\n')

    criterion = nn.CrossEntropyLoss()
    # 只把 fc 的参数交给优化器。backbone 冻结了，交给它也没用
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, EPOCH + 1):
        model.train()
        t0 = time.time()
        correct = total_n = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total_n += y.size(0)

        # 验证
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += y.size(0)

        val_acc = v_correct / v_total
        print(f'epoch:{epoch}, loss:{loss_sum / total_n:.4f}, '
              f'train acc:{correct / total_n:.4f}, val_acc:{val_acc:.4f}, '
              f'lr:{optimizer.param_groups[0]["lr"]:.5f}, time:{time.time() - t0:.0f}s')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f'\n最佳 val_acc: {best_acc:.4f}')
    print(f'权重已存: {SAVE_PATH}')
    print('立刻备份一份，防止下次跑别的实验把它覆盖了：')
    print(f'  cp {SAVE_PATH} best_transfer_{best_acc * 100:.2f}.pth')

    # ---------- 最终测试：官方 val 目录 1 万张，全程只看一次 ----------
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()
    test_set = TinyImageNetVal(transform=val_tf)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    t_correct = t_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            t_correct += (out.argmax(1) == y).sum().item()
            t_total += y.size(0)

    print(f'test acc: {t_correct / t_total:.4f}')


if __name__ == '__main__':
    main()

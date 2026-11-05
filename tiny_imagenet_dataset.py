import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
DATA_ROOT = r"C:\Users\wjt\WorkBuddy\2026-10-29-16-12-57\tiny-imagenet-project\dataset\tiny-imagenet-200"
train_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                         std=[0.2770, 0.2691, 0.2821]),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                         std=[0.2770, 0.2691, 0.2821]),
])

class TinyImageNetTrain(Dataset):
    """训练集：200个类别文件夹，每个500张图"""

    def __init__(self, root=DATA_ROOT, transform=train_transform):
        self.transform = transform
        self.samples = []                      # 存 [图片完整路径, 类别编号] 的列表

        # os.listdir 列出 train/ 下所有类别文件夹名，sorted 保证编号稳定
        wnid_dirs = sorted(os.listdir(os.path.join(root, 'train')))

        # 建立 "类别文件夹名 → 0~199 的数字标签" 的字典
        # 神经网络输出层只认数字，不认 'n01443537' 这种字符串
        self.wnid_to_label = {wnid: i for i, wnid in enumerate(wnid_dirs)}

        for wnid in wnid_dirs:
            img_dir = os.path.join(root, 'train', wnid, 'images')
            for img_name in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_name)
                self.samples.append((img_path, self.wnid_to_label[wnid]))

    def __len__(self):
        """DataLoader 会调这个方法问：你一共有多少条数据？"""
        return len(self.samples)

    def __getitem__(self, idx):
        """DataLoader 会调这个方法取第 idx 条数据，必须返回 (图片张量, 标签)"""
        img_path, label = self.samples[idx]
        # PIL.Image.open 是"惰性加载"，这里才真正读硬盘
        img = Image.open(img_path).convert('RGB')   # 统一转3通道（个别图是灰度图，不转会崩）
        if self.transform is not None:
            img = self.transform(img)               # 套用增强/预处理
        return img, label


# ============ 4. 验证集 Dataset 类（练手重点：标注在txt里，要自己解析） ============
class TinyImageNetVal(Dataset):
    """验证集：1万张图混在一个文件夹，类别要看 val_annotations.txt"""

    def __init__(self, root=DATA_ROOT, transform=val_transform):
        self.transform = transform
        self.samples = []

        # 类别编号表必须和训练集完全一致！用训练集那套映射
        wnid_dirs = sorted(os.listdir(os.path.join(root, 'train')))
        self.wnid_to_label = {wnid: i for i, wnid in enumerate(wnid_dirs)}

        # 解析 val_annotations.txt：每行形如
        # "val_0.JPEG\tn03425413\t..."（图片名 + TAB + 类别 + TAB + 坐标...）
        ann_file = os.path.join(root, 'val', 'val_annotations.txt')
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name, wnid = parts[0], parts[1]
                img_path = os.path.join(root, 'val', 'images', img_name)
                self.samples.append((img_path, self.wnid_to_label[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ============ 5. 自测代码：直接运行本文件就能验证写没写对 ============
if __name__ == '__main__':
    train_set = TinyImageNetTrain()
    val_set = TinyImageNetVal()

    print(f'训练集大小: {len(train_set)} (应为 100000)')
    print(f'验证集大小: {len(val_set)}   (应为 10000)')
    print(f'类别数: {len(train_set.wnid_to_label)} (应为 200)')

    img, label = train_set[0]           # 取第一条数据试试
    print(f'第0张图: 形状={img.shape}, 标签={label}')
    # 期望输出: 形状=torch.Size([3, 64, 64])  ← 3通道 64x64
    print('✅ 数据集读取正常！可以接 DataLoader 训练了')

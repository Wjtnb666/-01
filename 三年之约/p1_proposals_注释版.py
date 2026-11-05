"""
这是 p1_proposals.py 的逐行注释版，用来对着读。
逻辑与原文件完全一致，只多了注释；第 38 行那个改成 10 的实验被注释掉了。
"""
# ============================================================
# 第 1 段：导入。把别人写好的工具箱搬进来，后面才能用里面的函数。
# ============================================================
import sys                       # 系统工具，用来读"命令行里传了什么参数"

import matplotlib                # 画图库
matplotlib.use("Agg")            # 指定"只存文件，不弹窗口"。不加这句可能卡住
import matplotlib.patches as patches   # patches 里有个画矩形的工具，后面画框用
import matplotlib.pyplot as plt        # plt 是画图的主入口
import numpy as np                     # 数组计算，处理那张 1000x4 的表
import torch                           # 深度学习框架
import torchvision                     # torch 的视觉扩展包，模型都在这里
from PIL import Image                  # PIL 是读图片的库
from torchvision.models.detection import (   # 从"检测模型"这个抽屉里拿两样东西
    FasterRCNN_ResNet50_FPN_Weights,         # 预训练权重的名字（别人训好的成果）
    fasterrcnn_resnet50_fpn,                 # 造模型的函数
)
from torchvision.transforms import functional as F   # 图片和张量互转的工具

# ============================================================
# 第 2 段：定图片路径和设备
# ============================================================
# 这行是三元表达式：如果命令行给了参数就用它，否则用后面这个默认路径
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\wjt\PycharmProjects\PythonProject\三年之约\资源\帅比.JPG"
OUT_PATH = "proposals_vis.png"        # 结果图存到哪

# 有显卡就用显卡，没有就用 CPU。cuda 就是显卡那套东西
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | {device}")

# ============================================================
# 第 3 段：造模型
# ============================================================
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT   # 指名要"官方默认"的那套权重
model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
#   fasterrcnn_resnet50_fpn(...)  造出这个检测模型
#   .to(device)                   把模型搬到显卡上
#   .eval()                       切成"考试模式"：不训练，只推理
#                                 （会影响 Dropout 和 BN 的行为，你之前调过 model.train/eval）

# 下面这行是把"最多留几个框"从 1000 改成 10，用来做对比实验。
# 想恢复默认行为，把它注释掉即可。
# model.rpn._post_nms_top_n["testing"] = 10

# ============================================================
# 第 4 段：读图，变成网络能吃的数字
# ============================================================
img = Image.open(IMG_PATH).convert("RGB")   # 打开图片；convert("RGB") 统一成红绿蓝三通道
w0, h0 = img.size                           # 记下原图宽高（.size 返回 (宽, 高)）
tensor = F.pil_to_tensor(img).float().div(255).to(device)
#   F.pil_to_tensor  图片 -> 张量，数值范围 0~255 的整数
#   .float()         整数变小数
#   .div(255)        全部除以 255，范围变成 0~1
#   .to(device)      搬到显卡上

# ============================================================
# 第 5 段：打印几个参数，方便对照
# ============================================================
print(f"\n原图尺寸       : {w0} x {h0}")
print(f"RPN 关键参数   : pre_nms_top_n={model.rpn._pre_nms_top_n['testing']}, "
      f"post_nms_top_n={model.rpn._post_nms_top_n['testing']}, "
      f"nms_thresh={model.rpn.nms_thresh}")
# pre_nms_top_n  : 去重之前，先保留多少个候选
# post_nms_top_n : 去重之后，最多留多少个
# nms_thresh     : 去重时的严格程度

# ============================================================
# 第 6 段：走网络，拿到候选框（核心的 5 行）
# ============================================================
with torch.no_grad():                    # 这层缩进里的计算都不记梯度，省显存、更快
    images, _ = model.transform([tensor], None)
    # transform 做的是预处理：把图缩放、填充到统一尺寸
    # 它返回两个东西：(处理好的图, 其他信息)。我们用不到第二个，用 _ 接住扔掉
    features = model.backbone(images.tensors)
    # backbone 是主干网络：看图，提取特征（边缘、纹理、形状），输出一堆数字
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    # 防错：有的 backbone 直接返回张量，有的返回字典。统一成字典，后面才不会报错
    proposals, _ = model.rpn(images, features, None)
    # rpn 就是"猜框"的部件：看着特征，猜出一堆可能有东西的位置
    # 返回的 proposals 是一个列表，里面装着那张 N 行 4 列的坐标表

# ============================================================
# 第 7 段：把框换算回原图坐标
# ============================================================
p = proposals[0].cpu().numpy()
# proposals 是个列表，[0] 取第一张图的结果
# .cpu() 从显卡搬回内存，.numpy() 变成 numpy 数组，这样后面才能随意算
ih, iw = images.image_sizes[0]           # 送出网络的图实际被缩放成了多大
sx, sy = w0 / iw, h0 / ih                # 横向、纵向各放大了几倍
p_orig = p * np.array([sx, sy, sx, sy], dtype=np.float32)
# 四个坐标分别乘对应的倍数，框就从"缩略图坐标"变回"原图坐标"

# ============================================================
# 第 8 段：统计这些框有多大
# ============================================================
areas = (p_orig[:, 2] - p_orig[:, 0]) * (p_orig[:, 3] - p_orig[:, 1])
# p_orig[:, 2] 表示"所有行的第 3 列"，即 x2；[:, 0] 是 x1
# (x2 - x1) * (y2 - y1) = 宽 * 高 = 面积
ratio = areas / (w0 * h0)                # 面积占整张图的比例

print(f"\n送入网络尺寸   : {iw} x {ih}  (短边被缩放到 800)")
print(f"proposals 数量 : {len(p)}")       # len 给的是行数，也就是框的个数
bw = p_orig[:, 2] - p_orig[:, 0]         # 每个框的宽
bh = p_orig[:, 3] - p_orig[:, 1]         # 每个框的高
print(f"框宽范围       : {bw.min():.0f} ~ {bw.max():.0f} px")
print(f"框高范围       : {bh.min():.0f} ~ {bh.max():.0f} px")
print(f"宽高比 范围    : {(bw / bh).min():.2f} ~ {(bw / bh).max():.2f}")
print(f"面积占比       : 中位数 {np.median(ratio):.4f} | "
      f"最小 {ratio.min():.5f} | 最大 {ratio.max():.4f}")

# 把框按面积分成四档，看各有多少个
for lo, hi, name in [(0, 0.005, "极小 <0.5%"), (0.005, 0.05, "小 0.5-5%"),
                     (0.05, 0.3, "中 5-30%"), (0.3, 1.01, "大 >30%")]:
    n = int(((ratio >= lo) & (ratio < hi)).sum())
    # (ratio >= lo) & (ratio < hi) 是一个 True/False 数组，.sum() 数出 True 有几个
    print(f"  {name:<12}: {n:>5} 个 ({n / len(p) * 100:>5.1f}%)")
    # {name:<12} 左对齐占 12 格；{n:>5} 右对齐占 5 格

# ============================================================
# 第 9 段：对比实验，改去重严格程度，看框的数量怎么变
# ============================================================
print("\n--- 实验：改 nms_thresh，看框数量怎么变 ---")
cap = model.rpn._post_nms_top_n["testing"]   # 记下上限，用来判断有没有被截断
for thr in [0.3, 0.5, 0.7, 0.9]:             # 依次试四个值
    model.rpn.nms_thresh = thr               # 改设置
    with torch.no_grad():
        props, _ = model.rpn(images, features, None)   # 用新设置再猜一次
    n = len(props[0])
    flag = "  <- 撞上限，实际框数被截断" if n >= cap else ""
    print(f"  nms_thresh={thr}  ->  {n} 个框{flag}")

model.rpn.nms_thresh = 0.5                   # 改回去，不影响后面的画图

# ============================================================
# 第 10 段：把框画到图上，存成 png
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 12 * h0 / w0), dpi=110)
# 建一张画布；figsize 按原图宽高比算，免得图被拉变形
ax.imshow(img)                               # 把原图铺上去
for box in p_orig[:100]:                     # 只画前 100 个框
    x1, y1, x2, y2 = box                     # 一行四个数，分别取出来
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="#1D9E75",
                                   linewidth=0.8, alpha=0.85))
    # Rectangle(左上角坐标, 宽, 高)：fill=False 表示只画边框不填充
ax.set_title(f"RPN proposals: top 100 of {len(p)}, sorted by objectness", fontsize=12)
ax.axis("off")                               # 关掉坐标轴刻度
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")   # 存文件
print(f"\n可视化已保存: {OUT_PATH}")

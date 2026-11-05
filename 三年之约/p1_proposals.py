"""
P1-W1 第一步：看懂 Faster R-CNN 的 proposals 是什么。

用途：不训练、不评测，只观察 RPN 在一个真实图像上到底产生了什么。

用法：
    python p1_proposals.py                     # 用默认图
    python p1_proposals.py 你的图.jpg          # 用指定图

产出：
    终端打印 —— proposals 数量、尺度分布、NMS 阈值对数量的影响
    proposals_vis.png —— 原图 + objectness 最高的 100 个框
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms import functional as F

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\wjt\PycharmProjects\PythonProject\三年之约\资源\帅比.JPG"
OUT_PATH = "proposals_vis.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | {device}")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
model.rpn._post_nms_top_n["testing"] = 1000



img = Image.open(IMG_PATH).convert("RGB")
w0, h0 = img.size
tensor = F.pil_to_tensor(img).float().div(255).to(device)

print(f"\n原图尺寸       : {w0} x {h0}")
print(f"RPN 关键参数   : pre_nms_top_n={model.rpn._pre_nms_top_n['testing']}, "
      f"post_nms_top_n={model.rpn._post_nms_top_n['testing']}, "
      f"nms_thresh={model.rpn.nms_thresh}")

with torch.no_grad():
    images, _ = model.transform([tensor], None)
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = {"0": features}
    proposals, _ = model.rpn(images, features, None)

p = proposals[0].cpu().numpy()
ih, iw = images.image_sizes[0]
sx, sy = w0 / iw, h0 / ih
p_orig = p * np.array([sx, sy, sx, sy], dtype=np.float32)

areas = (p_orig[:, 2] - p_orig[:, 0]) * (p_orig[:, 3] - p_orig[:, 1])
ratio = areas / (w0 * h0)

print(f"\n送入网络尺寸   : {iw} x {ih}  (短边被缩放到 800)")
print(f"proposals 数量 : {len(p)}")
bw = p_orig[:, 2] - p_orig[:, 0]
bh = p_orig[:, 3] - p_orig[:, 1]
print(f"框宽范围       : {bw.min():.0f} ~ {bw.max():.0f} px")
print(f"框高范围       : {bh.min():.0f} ~ {bh.max():.0f} px")
print(f"宽高比 范围    : {(bw / bh).min():.2f} ~ {(bw / bh).max():.2f}")
print(f"面积占比       : 中位数 {np.median(ratio):.4f} | "
      f"最小 {ratio.min():.5f} | 最大 {ratio.max():.4f}")

# ---- 逐框明细：把 w / h / ratio 摊开看 ----
print(f"\n原图面积 W x H : {w0} x {h0} = {w0 * h0} 像素")
print("\n--- 逐框明细（前 15 个）---")
print(f"{'#':>3} {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7} "
      f"{'w':>6} {'h':>6} {'area':>10} {'ratio':>8}")
for i in range(min(len(p_orig), 15)):
    x1, y1, x2, y2 = p_orig[i]
    print(f"{i:>3} {x1:>7.0f} {y1:>7.0f} {x2:>7.0f} {y2:>7.0f} "
          f"{bw[i]:>6.0f} {bh[i]:>6.0f} {areas[i]:>10.0f} {ratio[i] * 100:>7.2f}%")
if len(p_orig) > 15:
    print(f"... 共 {len(p_orig)} 个框，只显示前 15 个")

for lo, hi, name in [(0, 0.005, "极小 <0.5%"), (0.005, 0.05, "小 0.5-5%"),
                     (0.05, 0.3, "中 5-30%"), (0.3, 1.01, "大 >30%")]:
    n = int(((ratio >= lo) & (ratio < hi)).sum())
    print(f"  {name:<12}: {n:>5} 个 ({n / len(p) * 100:>5.1f}%)")

print("\n--- 实验：改 nms_thresh，看框数量怎么变 ---")
cap = model.rpn._post_nms_top_n["testing"]
for thr in [0.3, 0.5, 0.7, 0.9]:
    model.rpn.nms_thresh = thr
    with torch.no_grad():
        props, _ = model.rpn(images, features, None)
    n = len(props[0])
    flag = "  <- 撞上限，实际框数被截断" if n >= cap else ""
    print(f"  nms_thresh={thr}  ->  {n} 个框{flag}")

model.rpn.nms_thresh = 0.5

fig, ax = plt.subplots(1, 1, figsize=(12, 12 * h0 / w0), dpi=110)
ax.imshow(img)
for box in p_orig[:100]:
    x1, y1, x2, y2 = box
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="#1D9E75", linewidth=0.8, alpha=0.85))
ax.set_title(f"RPN proposals: top 100 of {len(p)}, sorted by objectness", fontsize=12)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"\n可视化已保存: {OUT_PATH}")

"""单张图检测：真正的正常写法。全部逻辑就一行 model([img])。"""
import sys

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms import functional as F

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\wjt\PycharmProjects\PythonProject\三年之约\资源\帅比.JPG"

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).eval()
names = weights.meta["categories"]          # 91 个 COCO 类别名

img = F.pil_to_tensor(Image.open(IMG_PATH).convert("RGB")).float().div(255)

with torch.no_grad():
    out = model([img])[0]                   # ← 就这一行，缩放/推理/坐标换算全包了

keep = out["scores"] > 0.5
print(f"模型输出 {len(out['boxes'])} 个框，其中 score > 0.5 的有 {int(keep.sum())} 个\n")
for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
    if score < 0.5:
        continue
    x1, y1, x2, y2 = [round(float(v)) for v in box]
    print(f"{names[label]:<12} 把握 {score:.2f}   框 ({x1},{y1})-({x2},{y2})")

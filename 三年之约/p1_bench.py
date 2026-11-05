"""对比：在小图上算再换算 vs 直接在原图上算。只测速度，不看精度。"""
import time

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms import functional as F

IMG = r"C:\Users\wjt\PycharmProjects\PythonProject\三年之约\资源\帅比.JPG"

props = torch.cuda.get_device_properties(0)
print(f"显卡: {props.name}  显存 {props.total_memory / 1024**3:.1f} GB")
print(f"原图: {Image.open(IMG).size[0]} x {Image.open(IMG).size[1]}\n")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
img = F.pil_to_tensor(Image.open(IMG).convert("RGB")).float().div(255).cuda()

for short in [800, 1600, 2316]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = fasterrcnn_resnet50_fpn(
        weights=weights, min_size=short, max_size=short * 2
    ).cuda().eval()

    try:
        with torch.no_grad():
            model([img])                      # 预热，不计时
            torch.cuda.synchronize()
            t0 = time.time()
            out = model([img])
            torch.cuda.synchronize()
            dt = time.time() - t0

        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"短边 {short:>4}:  {dt:6.2f} 秒   峰值显存 {peak:5.2f} GB   "
              f"输出框数 {len(out[0]['boxes'])}")
    except torch.cuda.OutOfMemoryError:
        print(f"短边 {short:>4}:  显存不足，跑不动")

    del model
    torch.cuda.empty_cache()

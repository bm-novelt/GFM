import timm
import torch
from pathlib import Path
from models.teacher import EfficientNetForSimMIM, SwinTransformerForSimMIM
import os

PROJECT_HOME = Path("/home/bm/project/GFM/")
WEIGHTS_PATH = Path("output/simmim_finetune/efficientnet_v2m_21k.pth")
save_path = os.path.join(PROJECT_HOME, WEIGHTS_PATH)
# Load the EfficientNetV2-M model pretrained on ImageNet-21k
#model = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=True)
model = SwinTransformerForSimMIM(num_classes = 0)
model.train()

# These are the indices from your error
unused_indices = set(range(648, 748))  # or just the exact list if not contiguous

img_size = 224

dummy = torch.randn(2, 3, img_size, img_size)
mask = torch.randint(0, 2, (2, 14, 14))  # Example low-res mask

x = model.patch_embed(dummy)
print(x.shape)
#dummy_loss.backward()
"""
print(any(p.requires_grad and p.grad is not None for p in model.parameters()))

with open("/home/bm/project/GFM/swin_model_structure.txt", "w") as f:
    print(model, file=f)

x = model.conv_stem(dummy)
x = model.bn1(x)
print(x.shape)


for blk in model.blocks:
    x = blk(x)
    print(x.shape)

x = model.conv_head(x)
x = model.bn2(x)
print(x.shape)
x = model.global_pool(x)
x = model. classifier(x)
print(x.shape)"""
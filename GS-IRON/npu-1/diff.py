import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

img1 = np.array(Image.open("data/mip-flower/npu/output0.png").convert("RGB"), dtype=int)
img2 = np.array(Image.open("../../../gaussian-splatting/output/mip-flowers/train/ours_30000/renders/output0.png").convert("RGB"), dtype=int)

diff = np.abs(img1 - img2)
diff = np.clip(diff * 5, 0, 255)  # 強調倍率
ssim = compare_ssim(img1, img2, channel_axis=-1,
    data_range=255)
print(f"SSIM: {ssim}")

Image.fromarray(diff.astype("uint8")).save("diff.png")
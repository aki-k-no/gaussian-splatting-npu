import numpy as np
from PIL import Image

img1 = np.array(Image.open("data/ficus/npu/output0.png").convert("RGB"), dtype=int)
img2 = np.array(Image.open("data/ficus/cpu/output0.png").convert("RGB"), dtype=int)

diff = np.abs(img1 - img2)
diff = np.clip(diff * 5, 0, 255)  # 強調倍率

Image.fromarray(diff.astype("uint8")).save("diff.png")
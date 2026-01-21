import os
import re
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def compute_psnr(img1, img2, max_pixel=255.0):
    mse = mean_squared_error(img1.flatten(), img2.flatten())
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser(
        description="Compute average PSNR and SSIM for common outputN.png files"
    )
    parser.add_argument("-a", "--dir_a", required=True, help="比較元ディレクトリA")
    parser.add_argument("-b", "--dir_b", required=True, help="比較先ディレクトリB")
    args = parser.parse_args()

    dir_a = os.path.abspath(args.dir_a)
    dir_b = os.path.abspath(args.dir_b)

    pattern = re.compile(r"output(\d+)\.png")

    files_a = {f for f in os.listdir(dir_a) if True} #pattern.fullmatch(f)}
    files_b = {f for f in os.listdir(dir_b) if True} # pattern.fullmatch(f)}


    common_files = files_a & files_b

    if not common_files:
        print("共通ファイル数: 0")
        print("平均 PSNR: nan")
        print("平均 SSIM: nan")
        return

    psnr_values = []
    ssim_values = []

    for fname in common_files:
        img_a = load_image(os.path.join(dir_a, fname))
        img_b = load_image(os.path.join(dir_b, fname))

        if img_a.shape != img_b.shape:
            continue

        psnr_val = compute_psnr(img_a, img_b)
        if np.isfinite(psnr_val):
            psnr_values.append(psnr_val)

        ssim_val = ssim(
            img_a,
            img_b,
            channel_axis=2,
            data_range=255
        )
        if(ssim_val <= 0.8):
            print(f"警告: {fname} の SSIM が低い値 {ssim_val:.6f} です。")
        ssim_values.append(ssim_val)

    avg_psnr = np.mean(psnr_values) if psnr_values else float("nan")
    avg_ssim = np.mean(ssim_values) if ssim_values else float("nan")

    print("===== Summary =====")
    print(f"共通ファイル数 : {len(common_files)}")
    print(f"平均 PSNR      : {avg_psnr:.4f}")
    print(f"平均 SSIM      : {avg_ssim:.6f}")


if __name__ == "__main__":
    main()

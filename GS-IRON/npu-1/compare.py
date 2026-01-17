import argparse
import sys
from PIL import Image
import numpy as np
import os

#!/usr/bin/env python3
"""
Compare two images and compute the average of the squared distance (MSE).
Usage:
    python compare.py path/to/img1 path/to/img2 [--normalize]
"""

def compute_mse(img1_path: str, img2_path: str, normalize: bool = False) -> float:
    
    # If the provided paths are directories, compute the average MSE across matched image filenames.
    if os.path.isdir(img1_path) and os.path.isdir(img2_path):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        files1 = {f for f in os.listdir(img1_path) if os.path.splitext(f)[1].lower() in exts}



        files2 = {f for f in os.listdir(img2_path) if os.path.splitext(f)[1].lower() in exts}

        
        # if filename is like 00001.png, 00002.png, ...
        # convert it to output1.png, output2.png, ...
        # Check if filenames are numeric (like 00001.png) before converting
        # if any(os.path.splitext(f)[0].isdigit() for f in files1):
        #     files1 = {f"output{int(os.path.splitext(f)[0])}{os.path.splitext(f)[1]}" for f in files1}
        # if any(os.path.splitext(f)[0].isdigit() for f in files2):
        #     files2 = {f"output{int(os.path.splitext(f)[0])}{os.path.splitext(f)[1]}" for f in files2}

        common = sorted(files1 & files2)
        if not common:
            raise ValueError("No common image files found between the two directories.")

        mses = []
        maes = []
        for fname in common:
            p1 = os.path.join(img1_path, fname)
            p2 = os.path.join(img2_path, fname)
            mse, mae = compute_mse_one(p1, p2, normalize=normalize)
            mses.append(mse)
            maes.append(mae)

        return float(np.mean(mses)), float(np.mean(maes))
    
def compute_mse_one(img1_path: str, img2_path: str, normalize: bool = False) -> float:
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    if img1.size != img2.size:
        raise ValueError(f"Image sizes differ: {img1.size} vs {img2.size}.")
                

    a = np.asarray(img1, dtype=np.float32)
    b = np.asarray(img2, dtype=np.float32)
    if normalize:
        a /= 255.0
        b /= 255.0
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    return mse, mae

def compute_ssim(img1_path, img2_path, normalize: bool = False) -> float:
    def _gaussian_1d(size=11, sigma=1.5):
        x = np.arange(size) - (size - 1) / 2.0
        g = np.exp(-(x ** 2) / (2 * sigma * sigma))
        g /= g.sum()
        return g.astype(np.float32)

    def _sep_conv2d(img, k1d):
        # img: HxWxC float32, k1d: 1D kernel
        if img.ndim == 2:
            img = img[..., None]
        H, W, C = img.shape
        r = k1d.shape[0] // 2

        # Convolve along width
        padded_w = np.pad(img, ((0, 0), (r, r), (0, 0)), mode="reflect")
        tmp = np.empty_like(img, dtype=np.float32)
        for i in range(H):
            row = padded_w[i]
            for c in range(C):
                tmp[i, :, c] = np.convolve(row[:, c], k1d, mode="valid")

        # Convolve along height
        padded_h = np.pad(tmp, ((r, r), (0, 0), (0, 0)), mode="reflect")
        out = np.empty_like(img, dtype=np.float32)
        for j in range(W):
            col = padded_h[:, j, :]
            for c in range(C):
                out[:, j, c] = np.convolve(col[:, c], k1d, mode="valid")

        return out if out.shape[2] != 1 else out[:, :, 0]

    def _ssim_pair(p1, p2, normalize=False):
        a = Image.open(p1).convert("RGB")
        b = Image.open(p2).convert("RGB")
        if a.size != b.size:
            raise ValueError(f"Image sizes differ: {a.size} vs {b.size}.")
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if normalize:
            a *= (1.0 / 255.0)
            b *= (1.0 / 255.0)

        L = 1.0 if normalize else 255.0
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        k1d = _gaussian_1d(size=11, sigma=1.5)

        mu1 = _sep_conv2d(a, k1d)
        mu2 = _sep_conv2d(b, k1d)

        a2 = a * a
        b2 = b * b
        ab = a * b

        sigma1_sq = _sep_conv2d(a2, k1d) - mu1 * mu1
        sigma2_sq = _sep_conv2d(b2, k1d) - mu2 * mu2
        sigma12 = _sep_conv2d(ab, k1d) - mu1 * mu2

        # Numerical stability
        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / (den + 1e-12)

        return float(np.mean(ssim_map))

    if os.path.isdir(img1_path) and os.path.isdir(img2_path):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        set1 = {f for f in os.listdir(img1_path) if f.lower().endswith(exts)}
        set2 = {f for f in os.listdir(img2_path) if f.lower().endswith(exts)}
        
        # if any(os.path.splitext(f)[0].isdigit() for f in set1):
        #     set1 = {f"output{int(os.path.splitext(f)[0])}{os.path.splitext(f)[1]}" for f in set1}
        # if any(os.path.splitext(f)[0].isdigit() for f in set2):
        #     set2 = {f"output{int(os.path.splitext(f)[0])}{os.path.splitext(f)[1]}" for f in set2}

        common = sorted(set1.intersection(set2))
        if not common:
            raise ValueError("No matching image filenames across the two directories.")
        vals = []
        for name in common:
            val = _ssim_pair(os.path.join(img1_path, name), os.path.join(img2_path, name), normalize)
            if(val < 0.9):
                print(f"SSIM low for {name}: {val}")
            vals.append(val)
        return float(np.mean(vals))

    return _ssim_pair(img1_path, img2_path, normalize)
    # Placeholder for SSIM computation
    pass

def main():
        parser = argparse.ArgumentParser(description="Compute average squared distance (MSE) between two images.")
        parser.add_argument("dir1", help="First dir image path")
        parser.add_argument("dir2", help="Second dir image path")
        parser.add_argument("--normalize", action="store_true", help="Normalize pixel values to [0,1] before MSE")
        args = parser.parse_args()

        try:
                mse, mae = compute_mse(args.dir1, args.dir2, normalize=args.normalize)
                ssim = compute_ssim(args.dir1, args.dir2, normalize=args.normalize)
        except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"SSIM: {ssim:.6f}")

if __name__ == "__main__":
        main()
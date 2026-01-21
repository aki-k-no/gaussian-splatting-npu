import os
import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="ディレクトリ内のPNG画像の透明部分を白に置き換える")
    parser.add_argument("-d", "--dir", required=True, help="入力ディレクトリ")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.dir)
    output_dir = input_dir  # 出力用サブディレクトリ

    if not os.path.isdir(input_dir):
        print(f"Error: ディレクトリが存在しません: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".png"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 画像をRGBAで読み込む
        img = Image.open(input_path).convert("RGBA")

        # 白背景を作成
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

        # アルファブレンド
        img_with_white = Image.alpha_composite(white_bg, img)

        # RGBに変換して保存
        img_with_white.convert("RGB").save(output_path)
        count += 1

    print(f"{count} 枚のPNG画像を白背景に変換して {output_dir} に保存しました。")

if __name__ == "__main__":
    main()

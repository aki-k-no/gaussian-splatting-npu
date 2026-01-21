import os
import re
import shutil
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Copy numbered PNG files and rename them to outputN.png"
    )
    parser.add_argument(
        "-d", "--dir",
        required=True,
        help="基準ディレクトリへのパス"
    )

    args = parser.parse_args()
    base_dir = os.path.abspath(args.dir)

    if not os.path.isdir(base_dir):
        print(f"Error: ディレクトリが存在しません: {base_dir}", file=sys.stderr)
        sys.exit(1)

    source_dir = os.path.abspath(
        os.path.join(base_dir, "../../../test/ours_30000/render")
    )

    if not os.path.isdir(source_dir):
        print(f"Error: コピー元ディレクトリが存在しません: {source_dir}", file=sys.stderr)
        sys.exit(1)

    # ===== 既存 outputN.png の最大 N を取得 =====
    output_pattern = re.compile(r"output(\d+)\.png")
    max_index = 0

    for filename in os.listdir(base_dir):
        match = output_pattern.fullmatch(filename)
        if match:
            idx = int(match.group(1))
            max_index = max(max_index, idx)

    # ===== コピー元ファイル取得（昇順） =====
    source_files = sorted(
        f for f in os.listdir(source_dir)
        if re.fullmatch(r"\d+\.png", f)
    )

    if not source_files:
        print("Warning: コピー対象の png ファイルが見つかりません")
        return

    # ===== コピー & リネーム =====
    current_index = max_index + 1

    for src_file in source_files:
        src_path = os.path.join(source_dir, src_file)
        dst_name = f"output{current_index}.png"
        dst_path = os.path.join(base_dir, dst_name)

        shutil.copy2(src_path, dst_path)
        current_index += 1

    print(f"完了: {len(source_files)} ファイルをコピーしました")
    print(f"output{max_index + 1}.png 〜 output{current_index - 1}.png を作成")


if __name__ == "__main__":
    main()

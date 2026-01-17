# rename 00001.png to output1.png
import os
def rename_images_in_directory(directory: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if ext.lower() in exts and name.isdigit():
            new_name = f"output{int(name)}{ext}"
            os.rename(
                os.path.join(directory, filename),
                os.path.join(directory, new_name)
            )
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python rename.py path/to/directory")
        sys.exit(1)
    dir_path = sys.argv[1]
    rename_images_in_directory(dir_path)
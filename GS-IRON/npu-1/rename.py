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
    # get all subdirectories and apply
    for root, dirs, files in os.walk(dir_path):
        # since this directory is not the place we want, seek their subdirectory ./train/ours_30000/renders
        root2 = os.path.join(root, "train", "ours_30000", "renders")
        root3 = os.path.join(root, "train", "ours_30000", "gt")
        if os.path.exists(root2):
            rename_images_in_directory(root2)
        
        if os.path.exists(root3):
            rename_images_in_directory(root3)
import os
import shutil
from glob import glob

# ============================
# 수정할 부분
ROOT = "/home/jwlee/volume/Qwen2-vl-finetune-wo/data/ko_zh_datasets_4"
OUT = os.path.join(ROOT, "total_images")

src_folders = [
    os.path.join(ROOT, "train"),
    os.path.join(ROOT, "test"),
    os.path.join(ROOT, "val"),
]
# ============================

os.makedirs(OUT, exist_ok=True)

def copy_all_images(src):
    # jpg, png 전부 지원
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(src, "**", ext), recursive=True))

    print(f"[INFO] found {len(paths)} images in {src}")

    copied = 0

    for p in paths:
        fn = os.path.basename(p)  # 파일명만 남김
        dst = os.path.join(OUT, fn)

        if not os.path.exists(dst):  # 이미 있으면 skip
            shutil.copy(p, dst)
            copied += 1

    print(f"[INFO] copied={copied}, skipped={len(paths)-copied}")

if __name__ == "__main__":
    for folder in src_folders:
        copy_all_images(folder)

    print(f"[DONE] All images merged into: {OUT}")

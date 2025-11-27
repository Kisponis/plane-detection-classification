# yolo/prepare_coco_airplanes.py
import json
from pathlib import Path

import cv2
from pycocotools.coco import COCO
from tqdm import tqdm

COCO_ROOT = Path("data/coco")  # внутри: train2017, val2017, annotations/*
OUT_ROOT = Path("data/coco_airplanes")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

AIRPLANE_CAT_ID = 5  # COCO airplane

def convert_split(split: str):
    img_dir = COCO_ROOT / f"{split}2017"
    ann_path = COCO_ROOT / "annotations" / f"instances_{split}2017.json"

    out_img_dir = OUT_ROOT / "images" / split
    out_lbl_dir = OUT_ROOT / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(ann_path))

    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids, desc=f"{split}"):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[AIRPLANE_CAT_ID], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            continue

        src_img_path = img_dir / img_info["file_name"]
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # копируем изображение
        dst_img_path = out_img_dir / img_info["file_name"]
        if not dst_img_path.exists():
            cv2.imwrite(str(dst_img_path), img)

        # YOLO: class x_center y_center width height (нормированные)
        lines = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # у нас один класс -> id 0
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        label_path = out_lbl_dir / (Path(img_info["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(lines))


if __name__ == "__main__":
    convert_split("train")
    convert_split("val")

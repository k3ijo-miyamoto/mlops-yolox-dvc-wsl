import os
import json
import glob
from PIL import Image

def convert_yolo_to_coco(image_dir, label_dir, classes_file, output_json):
    # クラス読み込み
    with open(classes_file, "r") as f:
        class_names = [x.strip() for x in f.readlines()]

    images = []
    annotations = []
    categories = []
    annotation_id = 1

    for i, name in enumerate(class_names):
        categories.append({"id": i, "name": name, "supercategory": "none"})

    image_id = 1
    for img_path in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
        filename = os.path.basename(img_path)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            continue  # 画像が壊れている場合などをスキップ

        images.append({
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id
        })

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, x_center, y_center, w, h = map(float, parts)
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2

                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(cls_id),
                        "bbox": [x_min, y_min, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        image_id += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=2)

    print(f"[✓] COCO JSON 生成完了 → {output_json}")


# === 処理対象（train / val 両方） ===
base = "datasets/coco128"
convert_yolo_to_coco(
    image_dir=os.path.join(base, "images/train2017"),
    label_dir=os.path.join(base, "labels/train2017"),
    classes_file=os.path.join(base, "classes.txt"),
    output_json=os.path.join(base, "annotations/instances_train2017.json")
)

convert_yolo_to_coco(
    image_dir=os.path.join(base, "images/val2017"),
    label_dir=os.path.join(base, "labels/val2017"),
    classes_file=os.path.join(base, "classes.txt"),
    output_json=os.path.join(base, "annotations/instances_val2017.json")
)

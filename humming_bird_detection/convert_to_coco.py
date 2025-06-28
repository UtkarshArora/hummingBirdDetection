import os
import json
from PIL import Image

# Change this to match your path
IMAGE_DIR = './data/raw/Dataset_HBird/train/images'
LABEL_DIR = './data/raw/Dataset_HBird/train/lables'
OUTPUT_JSON = './data/raw/Dataset_HBird/train/annotations/hummingbirds.json'
CATEGORY_NAME = 'hummingbird'

images = []
annotations = []
categories = [{"id": 0, "name": CATEGORY_NAME}]
annotation_id = 0

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
image_files_len = len(image_files)

for image_id, filename in enumerate(sorted(image_files)):
    image_path = os.path.join(IMAGE_DIR, filename)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")

    with Image.open(image_path) as img:
        width, height = img.size

    images.append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    if not os.path.exists(label_path):
        continue  # No labels, skip

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, x_center, y_center, w, h = map(float, parts)
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            x = x_center - w / 2
            y = y_center - h / 2

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

# Final COCO structure

print(len(annotations))
coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_output, f, indent=4)

print(f"✅ COCO file saved to {OUTPUT_JSON}")

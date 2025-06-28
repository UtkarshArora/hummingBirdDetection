import os
import json
from PIL import Image
import random

IMAGE_DIR = "./data/raw/Dataset_HBird/train/images"
LABEL_DIR = "./data/raw/Dataset_HBird/train/labels"
OUTPUT_TRAIN_JSON = './data/raw/Dataset_HBird/train/annotations/trainData.json'
OUTPUT_TEST_JSON = './data/raw/Dataset_HBird/test/annotations/testData.json'
CATEGORY_NAME = 'hummingbird'

train_images = []
train_annotations = []
test_images = []
test_annotations = []
categories = [{"id": 0, "name": CATEGORY_NAME}]
annotation_id = 0

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files_len = len(image_files)

random.seed(42)
random.shuffle(image_files)

split_idx = int(image_files_len * 0.8)

train_filenames = image_files[:split_idx]
test_filenames = image_files[split_idx:]

# def yolo_to_coco_bbox(x_center_norm, y_center_norm, w_norm, h_norm, width, height):
#     # x_center = x_center_norm * width
#     # y_center = y_center_norm * height
#     x = (x_center_norm - w_norm / 2) * width
#     y = (y_center_norm - h_norm / 2) * height
#     w = w_norm * width
#     h = h_norm * height
#     return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]

def yolo_to_coco_bbox(x_center_norm, y_center_norm, w_norm, h_norm, width, height):
    x_center = x_center_norm * width
    y_center = y_center_norm * height
    w = w_norm * width
    h = h_norm * height
    x = x_center - w / 2
    y = y_center - h / 2
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


def process_split(filenames, image_list, annotation_list, starting_image_id):
    global annotation_id
    for local_image_id, filename in enumerate(sorted(filenames)):
        image_id = starting_image_id + local_image_id
        image_path = os.path.join(IMAGE_DIR, filename)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")

        with Image.open(image_path) as img:
            width, height = img.size

        image_list.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                bbox = yolo_to_coco_bbox(x_center, y_center, w, h, width, height)

                annotation_list.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [round(val, 2) for val in bbox],
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0
                })
                annotation_id += 1

process_split(train_filenames, train_images, train_annotations, starting_image_id=0)
process_split(test_filenames, test_images, test_annotations, starting_image_id=len(train_images))


os.makedirs(os.path.dirname(OUTPUT_TRAIN_JSON), exist_ok=True)
with open(OUTPUT_TRAIN_JSON, "w") as f:
    json.dump({
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }, f, indent=4)

# Write test.json
os.makedirs(os.path.dirname(OUTPUT_TEST_JSON), exist_ok=True)
with open(OUTPUT_TEST_JSON, "w") as f:
    json.dump({
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories
    }, f, indent=4)

print(f"✅ Done. Train: {len(train_images)} images | Test: {len(test_images)} images")

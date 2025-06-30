# from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrImageProcessor, TrainingArguments, Trainer
# from datasets import load_dataset, Dataset
# import json
# import os
# from PIL import Image

from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrImageProcessor, TrainingArguments, Trainer
from datasets import Dataset
import json
import os
from PIL import Image
import torch

# Fix path consistency - use absolute or consistent relative paths
BASE_DIR = "../data/raw/Dataset_HBird"
IMAGE_DIR = f"{BASE_DIR}/train/images"
TRAIN_ANNOTATIONS = f"{BASE_DIR}/train/annotations/trainData.json"
TEST_ANNOTATIONS = f"{BASE_DIR}/test/annotations/testData.json"

class RTDetrDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # batch is a list of dictionaries from our initial dataset
        pixel_values = []
        labels = []

        for example in batch:
            try:
                # Load image
                image = Image.open(example["image_path"]).convert("RGB")
                image_annotations = example["objects"]

                # Prepare annotations in the format the processor expects
                if not image_annotations["bbox"]:
                    # Handle images with no objects by creating a dummy annotation
                    h, w = image.size[1], image.size[0]
                    dummy_bbox = [w // 2 - 1, h // 2 - 1, w // 2 + 1, h // 2 + 1]
                    annotations_for_proc = {
                        "image_id": example["image_id"],
                        "annotations": [{"bbox": dummy_bbox, "category_id": 0, "area": 4, "iscrowd": 0}]
                    }
                else:
                    annotations_for_proc = {
                        "image_id": example["image_id"],
                        "annotations": [
                            {"bbox": bbox, "category_id": cat_id, "area": area, "iscrowd": iscrowd}
                            for bbox, cat_id, area, iscrowd in zip(
                                image_annotations["bbox"], image_annotations["category_id"],
                                image_annotations["area"], image_annotations["iscrowd"]
                            )
                        ]
                    }

                # Process image and annotations
                encoding = self.processor(images=image, annotations=annotations_for_proc, return_tensors="pt")

                # The processor adds a batch dimension, so we squeeze it
                pixel_values.append(encoding["pixel_values"].squeeze(0))
                labels.append(encoding["labels"][0])

            except Exception as e:
                print(f"Skipping corrupted sample {example.get('image_id', 'unknown')}: {e}")
                continue
        
        # If all samples in the batch are corrupted, return an empty dict
        if not pixel_values:
            return {}

        # Collate the processed items into a batch
        final_batch = {
            "pixel_values": torch.stack(pixel_values),
            "labels": labels
        }
        return final_batch


def load_coco_dataset(image_dir, annotation_file):
    """Load COCO format dataset"""
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    dataset_entries = []
    for image_id, image_info in images.items():
        image_path = os.path.join(image_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            continue

        image_annotations = annotations_by_image.get(image_id, [])
        objects = {'bbox': [], 'category_id': [], 'area': [], 'iscrowd': []}
        for ann in image_annotations:
            x, y, w, h = ann['bbox']
            objects['bbox'].append([x, y, x + w, y + h])
            objects['category_id'].append(0)
            objects['area'].append(ann.get('area', w * h))
            objects['iscrowd'].append(ann.get('iscrowd', 0))

        dataset_entries.append({
            'image_id': image_id,
            'image_path': image_path,
            'objects': objects
        })

    return dataset_entries


# --- 3. Load Datasets (Simpler now) ---
print("Loading training dataset...")
train_data = load_coco_dataset(IMAGE_DIR, TRAIN_ANNOTATIONS)
train_dataset = Dataset.from_list(train_data)

print("Loading test dataset...")
test_data = load_coco_dataset(IMAGE_DIR, TEST_ANNOTATIONS)
test_dataset = Dataset.from_list(test_data)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# --- 4. Initialize Processor, Config, and Model (Unchanged) ---
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

new_config = RTDetrConfig.from_pretrained(
    "PekingU/rtdetr_r50vd",
    num_labels=1,
    id2label={0: "hummingbird"},
    label2id={"hummingbird": 0}
)

model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd",
    config=new_config,
    ignore_mismatched_sizes=True
)

# --- 5. Instantiate the new Data Collator ---
data_collator = RTDetrDataCollator(processor=processor)

# --- 6. Training Arguments and Trainer (Apply the fix) ---
training_args = TrainingArguments(
    output_dir="./hummingbird_detection/outputs",
    per_device_train_batch_size=2, # Increased slightly for stability
    num_train_epochs=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    remove_unused_columns=False, # IMPORTANT: Keep original columns for the collator
    use_cpu=True,
    dataloader_num_workers=0,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"
)

# Initialize the Trainer with our custom data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator # Use the new collator class instance
)

print("Starting training...")
trainer.train()

import supervision as sv

# Paths
# Load COCO annotations
with open(TEST_ANNOTATIONS, "r") as f:
    coco = json.load(f)

# Build image_id to file_name mapping
image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
image_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

# Build image_id to annotations mapping
from collections import defaultdict
anns_per_image = defaultdict(list)
for ann in coco["annotations"]:
    anns_per_image[ann["image_id"]].append(ann)

# Evaluation
model.eval()
targets = []
predictions = []

for image_id, file_name in image_id_to_file.items():
    img_path = os.path.join(IMAGE_DIR, file_name)
    image = Image.open(img_path).convert("RGB")
    w, h = image_id_to_size[image_id]

    # Prepare ground truth
    gt_boxes = []
    gt_labels = []
    for ann in anns_per_image[image_id]:
        x, y, box_w, box_h = ann["bbox"]
        gt_boxes.append([x, y, x + box_w, y + box_h])
        gt_labels.append(ann["category_id"])
    targets.append(sv.Detections(xyxy=gt_boxes, class_id=gt_labels))

    # Model prediction
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=0.3
    )[0]

    pred_boxes = results["boxes"].cpu().numpy() if torch.is_tensor(results["boxes"]) else results["boxes"]
    pred_scores = results["scores"].cpu().numpy() if torch.is_tensor(results["scores"]) else results["scores"]
    pred_labels = results["labels"].cpu().numpy() if torch.is_tensor(results["labels"]) else results["labels"]
    predictions.append(sv.Detections(xyxy=pred_boxes, class_id=pred_labels, confidence=pred_scores))

# Compute mAP
mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets,
)
print(f"mAP@[.5:.95]: {mean_average_precision.map50_95:.3f}")
print(f"mAP@.5: {mean_average_precision.map50:.3f}")
print(f"mAP@.75: {mean_average_precision.map75:.3f}")

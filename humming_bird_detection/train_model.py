# # from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrImageProcessor, TrainingArguments, Trainer
# # from datasets import load_dataset, Dataset
# # import xjson
# # import os
# # from PIL import Image


from roboflow import Roboflow
rf = Roboflow(api_key="emHcbgLhITmU2KHvC6I7")
project = rf.workspace("humming-bird-detection").project("label-birdfeeder-camera-observations")
version = project.version(3)
dataset = version.download("coco")


from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrImageProcessor, TrainingArguments, Trainer
from datasets import Dataset
import json
import os
from PIL import Image
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#BASE_DIR = "./Label-Birdfeeder-Camera-Observations-3"
TRAIN_IMAGE_DIR = "./Label-Birdfeeder-Camera-Observations-3/train"

class RTDetrDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        
        pixel_values = []
        labels = []

        for example in batch:
            try:
                
                image = Image.open(example["image_path"]).convert("RGB")
                image_annotations = example["objects"]

                
                if not image_annotations["bbox"]:

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

                encoding = self.processor(images=image, annotations=annotations_for_proc, return_tensors="pt")

                pixel_values.append(encoding["pixel_values"].squeeze(0))
                labels.append(encoding["labels"][0])

            except Exception as e:
                print(f"Skipping corrupted sample {example.get('image_id', 'unknown')}: {e}")
                continue
        
        if not pixel_values:
            return {}

        final_batch = {
            "pixel_values": torch.stack(pixel_values),
            "labels": labels
        }
        return final_batch

# Path to original dataset
ANNOTATION_PATH = "./Label-Birdfeeder-Camera-Observations-3/train/_annotations.coco.json"
OUTPUT_DIR = "./Label-Birdfeeder-Camera-Observations-3"

with open(ANNOTATION_PATH, 'r') as f:
    coco = json.load(f)

random.seed(42)  # reproducibility
images = coco["images"]
random.shuffle(images)

split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
valid_images = images[split_idx:]

train_image_ids = {img["id"] for img in train_images}
valid_image_ids = {img["id"] for img in valid_images}
train_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in train_image_ids]
valid_annotations = [ann for ann in coco["annotations"] if ann["image_id"] in valid_image_ids]

split_coco = {
    "train": {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco["categories"]
    },
    "valid": {
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": coco["categories"]
    }
}

# Write new files
with open(os.path.join(OUTPUT_DIR, "train_split.json"), "w") as f:
    json.dump(split_coco["train"], f)

with open(os.path.join(OUTPUT_DIR, "valid_split.json"), "w") as f:
    json.dump(split_coco["valid"], f)

print("✅ Dataset split complete: train_split.json and valid_split.json created.")



def load_coco_dataset(image_dir, annotation_file):
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
            objects['category_id'].append(ann["category_id"])
            objects['area'].append(ann.get('area', w * h))
            objects['iscrowd'].append(ann.get('iscrowd', 0))

        dataset_entries.append({
            'image_id': image_id,
            'image_path': image_path,
            'objects': objects
        })

    return dataset_entries


TRAIN_ANNOTATIONS = "./Label-Birdfeeder-Camera-Observations-3/train_split.json"
TEST_ANNOTATIONS = "./Label-Birdfeeder-Camera-Observations-3/valid_split.json"

print("Loading training dataset...")
train_data = load_coco_dataset(TRAIN_IMAGE_DIR, TRAIN_ANNOTATIONS)
train_dataset = Dataset.from_list(train_data)

print("Loading test dataset...")
test_data = load_coco_dataset(TRAIN_IMAGE_DIR, TEST_ANNOTATIONS)
test_dataset = Dataset.from_list(test_data)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


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
model.to(device)


data_collator = RTDetrDataCollator(processor=processor)

training_args = TrainingArguments(
    output_dir="./hummingbird_detection/outputs",
    per_device_train_batch_size=2, 
    num_train_epochs=20,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    save_total_limit=3,
    load_best_model_at_end=True,
    eval_strategy="steps",
    eval_steps=500,  
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator 
)

print("Starting training...")

from pathlib import Path

checkpoint_dir = Path(training_args.output_dir)
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
if checkpoints:
    last_ckpt = str(checkpoints[-1])
    print(f"Resuming training from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    print("No checkpoint found. Starting fresh training...")
    trainer.train()

import supervision as sv

with open(TEST_ANNOTATIONS, "r") as f:
    coco = json.load(f)


image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
image_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

from collections import defaultdict
anns_per_image = defaultdict(list)
for ann in coco["annotations"]:
    anns_per_image[ann["image_id"]].append(ann)

model.eval()
targets = []
predictions = []

for image_id, file_name in image_id_to_file.items():
    img_path = os.path.join(TRAIN_IMAGE_DIR, file_name)
    image = Image.open(img_path).convert("RGB")
    w, h = image_id_to_size[image_id]


    gt_boxes = []
    gt_labels = []
    for ann in anns_per_image[image_id]:
        x, y, box_w, box_h = ann["bbox"]
        gt_boxes.append([x, y, x + box_w, y + box_h])
        gt_labels.append(ann["category_id"])
    targets.append(sv.Detections(xyxy=gt_boxes, class_id=gt_labels))

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=0.3
    )[0]

    pred_boxes = results["boxes"].cpu().numpy() if torch.is_tensor(results["boxes"]) else results["boxes"]
    pred_scores = results["scores"].cpu().numpy() if torch.is_tensor(results["scores"]) else results["scores"]
    pred_labels = results["labels"].cpu().numpy() if torch.is_tensor(results["labels"]) else results["labels"]
    predictions.append(sv.Detections(xyxy=pred_boxes, class_id=pred_labels, confidence=pred_scores))


mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions,
    targets=targets,
)
print(f"mAP@[.5:.95]: {mean_average_precision.map50_95:.3f}")
print(f"mAP@.5: {mean_average_precision.map50:.3f}")
print(f"mAP@.75: {mean_average_precision.map75:.3f}")



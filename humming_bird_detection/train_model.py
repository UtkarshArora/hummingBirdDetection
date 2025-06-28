from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrImageProcessor
from datasets import load_dataset

IMAGE_DIR = "./data/raw/Dataset_HBird/train/images"
train_dataset = load_dataset("coco", data_files = './data/raw/Dataset_HBird/train/annotations/trainData.json',
                              image_dir = IMAGE_DIR, 
                              split = "train")

processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

new_config = RTDetrConfig.from_pretrained(
    "PekingU/rtdetr_r50vd",
    num_labels=1,
    id2label={0: "hummingbird"},
    label2id={"hummingbird": 0}
)

def transform(example):
    image = example["image"]
    annotations = {"image_id": example["image_id"], "annotations": example["objects"]}
    encoding = processor(images=image, annotations=annotations, return_tensors="pt")
    return {
        "pixel_values": encoding["pixel_values"].squeeze(0),
        "pixel_mask": encoding["pixel_mask"].squeeze(0),
        "labels": encoding["labels"][0]
    }

model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", 
            config=new_config, 
            ignore_mismatched_sizes=True)

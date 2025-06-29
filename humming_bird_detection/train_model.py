from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset

IMAGE_DIR = "./data/raw/Dataset_HBird/train/images"
train_dataset = load_dataset(path="imagefolder",
    data_files={
        "annotations": "../data/raw/Dataset_HBird/train/annotations/trainData.json"
    },
    split="train",
    image_dir="../data/raw/Dataset_HBird/train/images")

test_dataset = load_dataset(path="imagefolder",
    data_files={
        "annotations": "../data/raw/Dataset_HBird/train/annotations/testData.json"
    },
    split="train",
    image_dir="../data/raw/Dataset_HBird/train/images")

processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

new_config = RTDetrConfig.from_pretrained(
    "PekingU/rtdetr_r50vd",
    num_labels=1,
    id2label={0: "hummingbird"},
    label2id={"hummingbird": 0}
)
# dataset = load_dataset('json', data_files='my_file.json')

def transform(example):
    image = example["image"]
    annotations = {"image_id": example["image_id"], "annotations": example["objects"]}
    encoding = processor(images=image, annotations=annotations, return_tensors="pt")
    return {
        "pixel_values": encoding["pixel_values"].squeeze(0),
        "pixel_mask": encoding["pixel_mask"].squeeze(0),
        "labels": encoding["labels"]
    }

model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", 
            config=new_config, 
            ignore_mismatched_sizes=True)

train_dataset = train_dataset.with_transform(transform)
test_dataset = test_dataset.with_transform(transform)

training_args = TrainingArguments(output_dir="./hummning_bird_detection/outputs",
                                per_device_train_batch_size=1,                 
                                num_train_epochs=1,                           
                                learning_rate=5e-5,
                                weight_decay=0.01,
                                logging_steps=10,
                                save_strategy="epoch",
                                evaluation_strategy="epoch",
                                remove_unused_columns=False,
                                no_cuda=True
                            )

trainer = Trainer(model = model , 
                  args = training_args,
                  train_dataset = train_dataset,
                  eval_dataset = test_dataset
                  )

trainer.train()
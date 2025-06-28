import torch
import requests

from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrConfig, RTDetrImageProcessor

file_path = './data/raw/Dataset_HBird/train/images/01_WSCT0001_JPG.rf.87c2095ce6af3ac18976871ad484ceb4.jpg'
image = Image.open(file_path)

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

new_config = RTDetrConfig.from_pretrained(
    "PekingU/rtdetr_r50vd",
    num_labels=1,
    id2label={0: "hummingbird"},
    label2id={"hummingbird": 0}
)

model = RTDetrForObjectDetection.from_pretrained( "PekingU/rtdetr_r50vd",
    config=new_config,
    ignore_mismatched_sizes=True )

inputs = image_processor(images=image, return_tensors="pt")

print(len(inputs))

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3)

print(len(results))
# for result in results:
#     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
#         score, label = score.item(), label_id.item()
#         box = [round(i, 2) for i in box.tolist()]
#         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
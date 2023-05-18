import ultralytics
import os
from pathlib import Path
import numpy as np
from PIL import Image
'''In this python file, only what you need to change are path to labels, model and images'''
path_to_model = 'yolov8n.pt'  # path to podel
path_to_img = './training/images'  # path to images
path_to_label = './training/labels'  # path to labels
model = ultralytics.YOLO('best.pt')
for file in os.listdir(path_to_img):
    img = Image.open(f'{path_to_img}/{file}')
    size = img.size
    results = model(img)
    file_only_path = Path(f'{path_to_img}/{file}').stem
    boxes = results[0].boxes
    for box in boxes:
        b = np.squeeze(box.data.cpu().numpy())
        a = np.squeeze(box.xywh.cpu().numpy())
        with open(f'{path_to_label}/{file_only_path}.txt', 'a') as photo:
            text = f'{int(b[-1])} {a[0] / size[0]} {a[1] / size[1]} {a[2] / size[0]} {a[3] / size[1]}\n'
            photo.write(text)

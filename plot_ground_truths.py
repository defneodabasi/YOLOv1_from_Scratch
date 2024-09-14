# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:02:47 2024

@author: defne
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

from train import Compose

class_dict = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
    5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable',
    11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant',
    16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
}

# Hyperparameters etc.
csv_file = "/home/defne-odabasi/PycharmProjects/yolov1_project/5_images_per_class.csv"
annotations = pd.read_csv(csv_file)
img_dir = "/home/defne-odabasi/PycharmProjects/yolov1_project/archive/images"
label_dir = "/home/defne-odabasi/PycharmProjects/yolov1_project/archive/labels"

def plot_ground_truth(image_path, label_path, img_name):
    boxes = []
    with open(label_path) as f:
        for label in f.readlines():
            class_label, x, y, width, height = [
                float(x) if float(x) != int(float(x)) else int(x)
                for x in label.replace("\n", "").split()
            ]
            boxes.append([class_label, x, y, width, height])
    image = Image.open(image_path)
    boxes = torch.tensor(boxes)
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    image, boxes = transform(image, boxes)

    # Convert to Cells
    S = 7
    C = 20
    B = 2
    label_matrix = torch.zeros((S, S, C + 5 * B))
    for box in boxes:
        class_label, x, y, width, height = box.tolist()
        class_label = int(class_label)
        i, j = int(S * y), int(S * x)
        x_cell, y_cell = S * x - j, S * y - i
        width_cell, height_cell = (width * S, height * S)
        if label_matrix[i, j, 20] == 0:
            label_matrix[i, j, 20] = 1
            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
            label_matrix[i, j, 21:25] = box_coordinates
            label_matrix[i, j, class_label] = 1

    bboxes1 = label_matrix[..., 21:25]
    bboxes2 = label_matrix[..., 26:30]
    scores = torch.cat(
        (label_matrix[..., 20].unsqueeze(0), label_matrix[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(1, 7, 1).unsqueeze(-1)
    x = 1 / 7 * (best_boxes[..., :1] + cell_indices)
    y = 1 / 7 * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / 7 * best_boxes[..., 2:4]
    w_h = w_h.reshape(1, 7, 7, 2)
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = label_matrix[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(label_matrix[..., 20], label_matrix[..., 25]).unsqueeze(-1)
    converted_labels = torch.cat(
        (predicted_class, best_confidence, converted_bboxes.squeeze()), dim=-1
    )

    bounding_boxes = []
    for i in range(S):
        for j in range(S):
            cell = converted_labels[i, j]
            predicted_class = cell[0].item()
            confidence = cell[1].item()
            x, y, w, h = cell[2:].tolist()
            if confidence > 0:
                bounding_boxes.append([predicted_class, confidence, x, y, w, h])

    fig, ax = plt.subplots(1)
    image = image.permute(1, 2, 0)
    height, width, _ = image.shape
    ax.imshow(image)
    for box in bounding_boxes:
        class_name = class_dict[int(box[0])]
        confidence = box[1]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        box_width = box[2] * width
        box_height = box[3] * height
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box_width,
            box_height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.text(
            upper_left_x * width,
            upper_left_y * height - 5,
            f"{class_name} {confidence:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='red')
        )
        ax.add_patch(rect)
        plt.axis('off')
    plt.suptitle(f"Ground Truth Image: {img_name}")
    plt.show()

for idx in range(len(annotations)):
    img_name = annotations.iloc[idx, 0]
    label_name = annotations.iloc[idx, 1]
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, label_name)
    plot_ground_truth(img_path, label_path, img_name)

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:20:45 2024

@author: defne
"""
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (intersection_over_union,
                   non_max_suppression,
                   mean_average_precision,
                   cellboxes_to_boxes,
                   get_bboxes,
                   plot_image,
                   save_checkpoint,
                   load_checkpoint,
                    train_plot
                   )

from loss import YoloLoss
import matplotlib.pyplot as plt
import time
import datetime

torch.cuda.empty_cache()
seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 3.5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "gpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 50
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
# LOAD_MODEL_FILE = "/home/defne-odabasi/PycharmProjects/yolov1_project/1_per_class.pth.tar"
IMG_DIR = "/home/defne-odabasi/PycharmProjects/yolov1_project/archive/images"
LABEL_DIR = "/home/defne-odabasi/PycharmProjects/yolov1_project/archive/labels"
NUM_CLASSES = 20
IOU_THRESHOLD = 0.4
THRESHOLD = 0.51
SAVEDIR = "/home/defne-odabasi/PycharmProjects/yolov1_project/archive/"
OUTPUT_FILENAME = "/home/defne-odabasi/PycharmProjects/yolov1_project/archive/train_progress_1"

S, B, C = 7, 2, NUM_CLASSES
# %%

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize(mean = [0.5], std = [0.5])])
transform = Compose([transforms.Resize((448, 448)),
                     transforms.ColorJitter(brightness=1.5, saturation=1.5),
                     transforms.ToTensor()])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss=loss.item())

    ml = sum(mean_loss) / len(mean_loss)
    print(f'Training loss: {ml}')
    return ml


# Used for validation function
def test_fn(valid_loader, model, loss_fn):
    model.eval()
    loop = tqdm(valid_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x,y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())

        # Updating the progress bar
        loop.set_postfix(loss=loss.item())

    pred_boxes, target_boxes = get_bboxes(valid_loader, model, iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD,
                                          num_classes=NUM_CLASSES)

    mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=IOU_THRESHOLD,
                                           box_format="midpoint", num_classes=NUM_CLASSES)

    ml = sum(mean_loss) / len(mean_loss)
    print(f"Validation loss :{ml}")
    print(f"Validation mAP: {mean_avg_prec}")
    model.train()
    return ml, mean_avg_prec


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    early_stopper = EarlyStopping(patience=5, min_delta=10)

    train_dataset = VOCDataset("/home/defne-odabasi/PycharmProjects/yolov1_project/train_set.csv",
                               transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

    valid_dataset = VOCDataset("/home/defne-odabasi/PycharmProjects/yolov1_project/val_set.csv",
                               transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

    test_dataset = VOCDataset("/home/defne-odabasi/PycharmProjects/yolov1_project/test_set.csv",
                              transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    train_map_list = []
    train_ml_list = []
    valid_map_list = []
    valid_ml_list = []

    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)

        train_mean_loss = train_fn(train_loader, model, optimizer, loss_fn)
        train_ml_list.append(train_mean_loss)

        valid_mean_loss, valid_map = test_fn(valid_loader, model, loss_fn)
        valid_ml_list.append(valid_mean_loss)
        valid_map_list.append(valid_map)

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD,
                                              num_classes=NUM_CLASSES)

        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=IOU_THRESHOLD,
                                               box_format="midpoint", num_classes=NUM_CLASSES)
        train_map_list.append(mean_avg_prec)
        print(f"Train mAP: {mean_avg_prec}")

        if early_stopper.early_stop(valid_mean_loss):
            break

        # Plotting:
        # for x, y in train_loader:
        #     x = x.to(DEVICE)
        #     # if epoch == 15:
        #     #     print("We are here")
        #     # Run a forward pass and print the model output
        #     outputs = model(x)
        #     for idx in range(1):
        #         bboxes = cellboxes_to_boxes(outputs, S=7, C=NUM_CLASSES)
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD,
        #                                      box_format="midpoint")
        #         plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes, savedir=SAVEDIR, epoch=epoch, S=7)
        #     # import sys
        #     # sys.exit()

    # saving the model
    if SAVE_MODEL:
        # Generate a timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_filename = f"{timestamp}-e{EPOCHS}-myyolov1-lr{LEARNING_RATE:.0e}-b{BATCH_SIZE}.pt"
        model_path = os.path.join('/home/defne-odabasi/PycharmProjects/yolov1_project/', model_filename)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=model_path)
        print(f"Model saved as {model_filename}")
        time.sleep(10)

    # PLOTTING
    plt.figure(figsize=(12, 5))
    # Loss Plot
    plt.plot(range(EPOCHS), train_ml_list, marker='o', label='Train Loss')
    plt.plot(range(EPOCHS), valid_ml_list, marker='o', label='Valid Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(f'{SAVEDIR}/loss.png', bbox_inches='tight')

    # Precision Plot
    plt.figure(figsize=(12, 5))
    plt.plot(range(EPOCHS), train_map_list, marker='o', label='Train mAP')
    plt.plot(range(EPOCHS), valid_map_list, marker='o', label='Valid mAP')
    plt.title('Training mAP vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.grid()
    plt.legend()
    plt.savefig(f'{SAVEDIR}/mAP.png', bbox_inches='tight')
    plt.show()

    model.eval()
    for x, labels in train_loader:
        x = x.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            predictions = model(x)

            # predictions = predictions.view(-1, S, S, 5 * B + C)
            #
            # # Apply softmax to class probabilities
            # class_probs = F.softmax(predictions[..., :C], dim=-1)
            # # class_probs = torch.sigmoid(predictions[..., :C])
            #
            # # Apply sigmoid to confidence scores
            # confidence1 = torch.sigmoid(predictions[..., C]).unsqueeze(-1)
            # confidence2 = torch.sigmoid(predictions[..., C + 5]).unsqueeze(-1)
            #
            # # # Check if the sum of class probabilities is equal to 1 for every grid cell
            # # class_probs_sum = class_probs.sum(dim=-1)
            # # if not torch.allclose(class_probs_sum, torch.ones_like(class_probs_sum)):
            # #     print("Error: Sum of class probabilities is not equal to 1 for some grid cells.")
            #
            # # # Check if confidence scores are bounded between 0 and 1
            # # if not (torch.all(confidence1 >= 0) and torch.all(confidence1 <= 1)):
            # #     print("Error: Confidence score 1 is not bounded between 0 and 1.")
            # # if not (torch.all(confidence2 >= 0) and torch.all(confidence2 <= 1)):
            # #     print("Error: Confidence score 2 is not bounded between 0 and 1.")
            #
            # # Bounding box coordinates don't need activation, they are already affected by linear activation
            # bbox1 = predictions[..., C + 1:C + 5]
            # bbox2 = predictions[..., C + 6:C + 10]
            #
            # # Concatenate everything back together
            # output = torch.cat((class_probs, confidence1, bbox1, confidence2, bbox2), dim=-1)
            #
            # output = output.view(-1, S * S * (5 * B + C))

        for i in range(1):
            labels_bboxes = torch.tensor(cellboxes_to_boxes(labels, S=7, C=NUM_CLASSES))
            bboxes = cellboxes_to_boxes(predictions, S=7, C=NUM_CLASSES)
            bboxes = non_max_suppression(bboxes[i], iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD,
                                         box_format="midpoint")
            # plot_image(x[i].permute(1, 2, 0).to("cpu"), bboxes, savedir=SAVEDIR, epoch=EPOCHS)
            train_plot(x[i].permute(1, 2, 0).to("cpu"), bboxes, labels_bboxes[i], epoch=EPOCHS, S=7, C=2)
    model.train()

if __name__ == "__main__":
    main()

# Creating Subplots from the images

# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# axs = axs.flatten()
#
# epochs_to_plot = [0, EPOCHS // 2, 2 * EPOCHS // 5, 3 * EPOCHS // 5, 4 * EPOCHS // 5, EPOCHS - 1]
# for i, epoch in enumerate(epochs_to_plot):
#     filename = os.path.join(SAVEDIR, f'epoch_{epoch}.png')
#     if os.path.exists(filename):
#         image = Image.open(filename)
#         axs[i].imshow(image)
#         axs[i].set_title(f'Epoch {epoch}')
#         axs[i].axis('off')
#     else:
#         print("Image for epoch {epoch} not found")
#
#     plt.suptitle("Training results for epochs")
#     plt.tight_layout()
#     plt.savefig(OUTPUT_FILENAME)
# # plt.show()


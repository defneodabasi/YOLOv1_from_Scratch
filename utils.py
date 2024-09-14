# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:32:36 2024

@author: defne
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# this is
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    global box2_x1
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        # print(f"Processing class {c}")
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        # print(f"Detections for class {c}: {len(detections)}")
        # print(f"Ground truths for class {c}: {len(ground_truths)}")
        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # print(f"Amount of bboxes before torch.zeros: {len(amount_bboxes)}")

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # amount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)  # sorting with respect to their confidence values
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0
            best_gt_idx = None

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou is not None and best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
        # accuracy as well
    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes, savedir, epoch, S=7):
    """Plots predicted bounding boxes on the image"""

    class_dict = {
        0: 'aeroplane',
        1: 'bicycle',
        2: 'bird',
        3: 'boat',
        4: 'bottle',
        5: 'bus',
        6: 'car',
        7: 'cat',
        8: 'chair',
        9: 'cow',
        10: 'diningtable',
        11: 'dog',
        12: 'horse',
        13: 'motorbike',
        14: 'person',
        15: 'pottedplant',
        16: 'sheep',
        17: 'sofa',
        18: 'train',
        19: 'tvmonitor'
    }

    im = np.array(image)
    height, width, _ = im.shape

    # print(f"Image size: {width}, {height}")

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # for i in range(S):
    #     # Vertical lines
    #     plt.axvline(x=i * width / S, color='yellow', linestyle='--', linewidth=0.5)
    #     # Horizontal lines
    #     plt.axhline(y=i * height / S, color='yellow', linestyle='--', linewidth=0.5)

    # Create a Rectangle patch
    for box in boxes:
        class_name = class_dict[int(box[0])]
        confidence = box[1]
        box = box[2:]
        # print(f"Bounding box coordinates (x, y, w, h): {box[0]}, {box[1]}, {box[2]}, {box[3]}")

        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        box_width = abs(box[2] * width)
        box_height = abs(box[3] * height)

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
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.axis('off')

    plt.title(f"Epoch: {epoch}")
    plt.show()
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    # plt.savefig(f'{savedir}/epoch_{epoch}.png', bbox_inches='tight')

    # plt.show()


def get_bboxes(
        loader,
        model,
        iou_threshold,
        threshold,
        num_classes,
        pred_format="cells",
        box_format="midpoint",
        device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        # print("Batch_idx: ", batch_idx)
        # print("x: ", x.shape)
        # print("Labels: ", labels.shape)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels, S=7, C=num_classes)
        bboxes = cellboxes_to_boxes(predictions, S=7, C=num_classes)

        # print(f"Batch {batch_idx}, true_bboxes shape: {[len(t) for t in true_bboxes]}")
        # print(f"Batch {batch_idx}, bboxes shape: {[len(b) for b in bboxes]}")

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            # print(f"nms_boxes for batch {batch_idx}, idx {idx}: {nms_boxes}")

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            # print(f"true_bboxes[idx]: {true_bboxes[idx]}")
            # print(f"true_bboxes shape: {len(true_bboxes)}, example: {true_bboxes[0]}")
            # print(f"bboxes shape: {len(bboxes)}, example: {bboxes[0]}")

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                # print(box)
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


# This is tested and identified as working without any problems
def convert_cellboxes(predictions, S=7, C=20):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + 10)
    bboxes1 = predictions[..., C + 1:C + 5]
    bboxes2 = predictions[..., C + 6:C + 10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    # This part is where the x,y,w,h are considered relative to the image.
    # Image size is in between 0 and 1.
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))

    # w and h are scaled relative to the image.
    w_h = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_pred_class_prob = predictions[..., :C].max(-1, keepdim=True).values
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(-1)
    best_class_conf = best_confidence * best_pred_class_prob
    converted_preds = torch.cat(
        (predicted_class, best_class_conf, converted_bboxes), dim=-1
    )
    # print("convert_cellboxes function output")
    # Debugging: Print intermediate results
    # print("Converted bboxes:", converted_bboxes.shape)
    # print("Predicted class:", predicted_class.shape)
    # print("Best confidence:", best_confidence.shape)

    return converted_preds


def cellboxes_to_boxes(out, S=7, C=20):
    # print("Out shape: ", out.shape)
    converted_pred = convert_cellboxes(out, S, C).reshape(out.shape[0], S * S, -1)
    # print("converted predictions: ", converted_pred.shape)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    # print("All boxes: \n", len(all_bboxes))
    # print(len(all_bboxes[0]))

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def train_plot(image, pred_bboxes, label_bboxes, epoch, S=7, C=20):
    class_dict = {
        0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
        5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable',
        11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant',
        16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
    }

    label_bboxes = label_bboxes.reshape(S, S, C+4)

    ground_bboxes = []
    for i in range(S):
        for j in range(S):
            cell = label_bboxes[i, j]
            predicted_class = cell[0].item()
            confidence = cell[1].item()
            x, y, w, h = cell[2:].tolist()
            if confidence > 0:
                ground_bboxes.append([predicted_class, confidence, x, y, w, h])

    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for box in pred_bboxes:
        class_name = class_dict[int(box[0])]
        confidence = box[1]
        box = box[2:]
        # print(f"Bounding box coordinates (x, y, w, h): {box[0]}, {box[1]}, {box[2]}, {box[3]}")

        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        box_width = abs(box[2] * width)
        box_height = abs(box[3] * height)

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box_width,
            box_height,
            linewidth=1.5,
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
        # Add the patch to the Axes
        ax.add_patch(rect)

    for box in ground_bboxes:
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
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--"
        )
        ax.text(
            upper_left_x * width,
            upper_left_y * height + 15,
            f"{class_name}",
            #f"{class_name} {confidence:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='purple')
        )
        ax.add_patch(rect)

    plt.axis('off')
    plt.title(f"Epoch: {epoch}")

    plt.suptitle(f"Epoch {epoch}:")
    plt.show()
# %%

# # Test cases
# boxes_preds_midpoint = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.7, 0.7, 0.2, 0.2]], dtype=torch.float32)
# boxes_labels_midpoint = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.6, 0.6, 0.2, 0.2]], dtype=torch.float32)
#
# boxes_preds_corners = torch.tensor([[0.3, 0.3, 0.7, 0.7], [0.6, 0.6, 0.8, 0.8]], dtype=torch.float32)
# boxes_labels_corners = torch.tensor([[0.3, 0.3, 0.7, 0.7], [0.65, 0.65, 0.85, 0.85]], dtype=torch.float32)
#
# iou_midpoint = intersection_over_union(boxes_preds_midpoint, boxes_labels_midpoint, box_format="midpoint")
# iou_corners = intersection_over_union(boxes_preds_corners, boxes_labels_corners, box_format="corners")
#
# print("IoU with midpoint format:", iou_midpoint)
# print("IoU with corners format:", iou_corners)

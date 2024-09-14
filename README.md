# 24-internship-object-detection #

<h1 align="center">YOLOv1 from Scratch</h1>

The implementation of YOLO-v1 is from the paper [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) 
The codebase for this project is developed with guidance from the [YOLO-v1 from scratch](https://www.youtube.com/watch?v=n9_XyCGr-MI) tutorial.

### Clone the repository 

```bash
git clone https://github.com/obss/24-internship-object-detection.git
```

### Environment Setup

Create a virtual environment

```bash
conda create --name test_env python=3.9
```
Activate the virtual environment

```bash
conda activate test_env
```

Installing the required dependencies

```bash
pip install -r requirements.txt
```

### Dataset

The dataset used during training, validation and testing can be downloaded from [Kaggle](https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2)

### YOLO Overview

#### YOLOv1:

<img src="https://miro.medium.com/v2/resize:fit:574/1*15uBgdR3_rNZzx665Leang.jpeg" alt="Grid Input" width="500"/>

- Divides the input into grid cells (7x7 = 49 cells).
- If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
- Each grid cell predicts **B = 2** bounding boxes and confidence scores for those boxes.
- Each bounding box consists of **5 predictions**: `x`, `y`, `w`, `h`, and `confidence`. The `(x, y)` center is relative to the grid cell, and the width and height are relative to the whole image.
- Predicts **C class probabilities** for the object.
- Bounding boxes are filtered using **non-max suppression** to remove overlapping boxes, choosing the highest probability boxes.
- Processes the entire image in one pass, hence the name **"You Only Look Once."**

#### 1.2 Loss Function:

<img src="https://pylessons.com/media/Tutorials/YOLO-tutorials/YOLOv3-TF2-mnist/loss_function.png" alt="Loss Function" width="500"/>

- Loss from bounding box coordinate predictions is high.
- Loss from confidence predictions for boxes that do not contain objects is low.
- The square root of the bounding box width and height is taken so that small deviations in large boxes matter less than in small boxes.

#### 1.3 IOU (Intersection over Union):

- Calculates the intersection over union of two bounding boxes, measuring overlap between them.

<img src="https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png" alt="IOU" width="500"/> 

- Takes `boxes_preds` (tensor) as predicted bounding boxes and `boxes_labels` (tensor) as ground truth bounding boxes.
- Computes the coordinates of the bounding rectangle, the intersection region, and the union region. IOU is calculated as the intersection over union.

#### 1.4 Non-Max Suppression:

- Filters overlapping bounding boxes based on IoU and confidence scores.

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/graphic4.jpg" alt="Non-Max Suppression" width="500"/>

- Takes a list of proposal boxes `B`, confidence scores `S`, and overlap threshold `N` as input. Outputs a list of filtered proposals `D`.
- The proposal with the highest confidence score is selected, added to the final proposal list `D`, and removed from `B`.
- The IoU of this proposal with other proposals in `B` is calculated, and proposals with IoU greater than the threshold are removed.
- This process is repeated until no more proposals remain in `B`.

#### 1.5 Mean Average Precision (mAP):

- A metric used to evaluate object detection models like YOLO.

<img src="https://cdn.prod.website-files.com/660ef16a9e0687d9cc27474a/662c42677529a0f4e97e4f96_644aea65cefe35380f198a5a_class_guide_cm08.png" alt="mAP" width="500"/>

- mAP is based on submetrics like confusion matrix, IoU, recall, and precision.
- Confusion matrix attributes include True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
- IoU measures the overlap of predicted bounding boxes with ground truth, where higher IoU indicates better accuracy.



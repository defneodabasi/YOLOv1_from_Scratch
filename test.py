#test.py
# test.py
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from loss import YoloLoss
from utils import load_checkpoint, cellboxes_to_boxes, non_max_suppression, train_plot
import torch.nn.functional as F
import numpy as np
# Parameters (ensure these match the parameters in your main training script)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL_FILE = "/home/defne-odabasi/PycharmProjects/yolov1_project/1_per_class.pth.tar"
IMG_DIR = "/home/defne-odabasi/PycharmProjects/yolov1_project/archive/images"
LABEL_DIR = "/home/defne-odabasi/PycharmProjects/yolov1_project/archive/labels"
NUM_CLASSES = 20
IOU_THRESHOLD = 0.4
THRESHOLD = 0.4
S, B, C = 7, 2, NUM_CLASSES

np.random.seed(42)
#%%
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes
# Transforms
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

# Load model
model = Yolov1(split_size=7, num_boxes=2, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = YoloLoss()

load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

# Load test dataset
test_dataset = VOCDataset("/home/defne-odabasi/PycharmProjects/yolov1_project/1_images_per_class.csv",
                          transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                         pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

def test_fn(test_loader, model, loss_fn):
    model.eval()
    loop = tqdm(test_loader, leave=True)
    mean_loss = []
    #pred_boxes, target_boxes = [], []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())

        # Updating the progress bar
        loop.set_postfix(loss=loss.item())

    ml = sum(mean_loss) / len(mean_loss)
    print(f"Test loss: {ml}")
    model.train()
    return ml

test_mean_loss = test_fn(test_loader, model, loss_fn)
print(f"Test loss: {test_mean_loss}")

#Plotting the test results:
model.eval()
for x, labels in test_loader:
    x = x.to(DEVICE)
    labels = labels.to(DEVICE)
    with torch.no_grad():
        predictions = model(x)
        predictions = predictions.view(-1, S, S, 5 * B + C)

        # Apply softmax to class probabilities
        class_probs = F.softmax(predictions[..., :C], dim=-1)

        # Apply sigmoid to confidence scores
        confidence1 = torch.sigmoid(predictions[..., C]).unsqueeze(-1)
        confidence2 = torch.sigmoid(predictions[..., C + 5]).unsqueeze(-1)

        # Check if the sum of class probabilities is equal to 1 for every grid cell
        class_probs_sum = class_probs.sum(dim=-1)
        if not torch.allclose(class_probs_sum, torch.ones_like(class_probs_sum)):
            print("Error: Sum of class probabilities is not equal to 1 for some grid cells.")

        # Check if confidence scores are bounded between 0 and 1
        if not (torch.all(confidence1 >= 0) and torch.all(confidence1 <= 1)):
            print("Error: Confidence score 1 is not bounded between 0 and 1.")
        if not (torch.all(confidence2 >= 0) and torch.all(confidence2 <= 1)):
            print("Error: Confidence score 2 is not bounded between 0 and 1.")

        # Bounding box coordinates don't need activation, they are already affected by linear activation
        bbox1 = predictions[..., C + 1:C + 5]
        bbox2 = predictions[..., C + 6:C + 10]

        # Concetenate everthing back together
        output = torch.cat((class_probs, confidence1, bbox1, confidence2, bbox2), dim=-1)

        output = output.view(-1, S * S * (5 * B + C))
    for i in range(len(predictions)):
        labels_bboxes = torch.tensor(cellboxes_to_boxes(labels, S=7, C=NUM_CLASSES))
        bboxes = cellboxes_to_boxes(output, S=7, C=NUM_CLASSES)
        bboxes = non_max_suppression(bboxes[i], iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD,
                                     box_format="midpoint")
        train_plot(x[i].permute(1, 2, 0).to("cpu"), bboxes, labels_bboxes[i], epoch=1, S=7, C=2)

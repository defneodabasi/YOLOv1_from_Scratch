'''
This file is created to observe the number of instances exist for a class.
The distribution of the instances from classes can be crucial for the model.
'''


import csv
import os

csv_file_path = 'test_set.csv'
labels_dir = 'archive/labels/'

# Initialize the class labels
class_labels = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
    5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable',
    11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant',
    16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
}

# Dictionary to keep count of each class, initializing all to 0
class_counts = {i: 0 for i in range(20)}

# Function to process each label file
def process_label_file(label_file_name):
    label_file_path = os.path.join(labels_dir, label_file_name)
    with open(label_file_path, 'r') as label_file:
        for line in label_file:
            class_id = int(line.split()[0])  # Extract class id (first number in the line)
            class_counts[class_id] += 1      # Increment count for this class

# Read the train_set.csv file
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip header row if it exists
    for row in csv_reader:
        label_file_path = row[1]  # Assuming the label file path is in the second column
        process_label_file(label_file_path)

print("Class distribution: ")
for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id} ({class_labels[class_id]} : {count} occurances")

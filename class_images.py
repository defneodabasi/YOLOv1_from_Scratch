'''
This file is created to create a .csv file which will have the particular instances of a class is determined.
For instance,
For instance=5, a file with name {instance}_images_per_class.csv is created.
Inside the .csv file, there will be 5 photos of the aeroplane class, 5 photos of bicycle class and so on.
The photos will contain only one object.
'''

import os
import csv

labels_dir = 'archive/labels/'
images_dir = 'archive/images/'

# Initialize the class labels
class_labels = {
    0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
    5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable',
    11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant',
    16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
}

# Dictionary to store image-label pairs for each class
class_images = {i: [] for i in range(20)}

# Function to process each label file
def process_label_file(label_file_name, max_instances):
    label_file_path = os.path.join(labels_dir, label_file_name)
    with open(label_file_path, 'r') as label_file:
        lines = label_file.readlines()
        if len(lines) == 1:
            class_id = int(lines[0].split()[0])  # Extract the class id (first number in the line)
            if len(class_images[class_id]) < max_instances:
                img_file_name = label_file_name.replace('.txt', '.jpg')
                class_images[class_id].append((img_file_name, label_file_name))


# Function to write image-label pairs to a CSV file
def write_csv(file_name, data):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['img', 'label'])
        writer.writerows(data)

# Write CSV files for 5, 10, and 15 images per class
for max_instances in [1, 5, 10, 15]:
    class_images = {i: [] for i in range(20)}  # Reset the dictionary for each instance count
    for label_file_name in os.listdir(labels_dir):
        process_label_file(label_file_name, max_instances)
        if all(len(class_images[class_id]) >= max_instances for class_id in class_images):
            break  # Break if all classes have the required number of image-label pairs

    csv_data = []
    for class_id, images in class_images.items():
        csv_data.extend(images[:max_instances])
    write_csv(f'{max_instances}_images_per_class.csv', csv_data)

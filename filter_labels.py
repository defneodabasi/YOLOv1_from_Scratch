import os
import pandas as pd

# Define the class dictionary
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

# Define the paths
label_dir = '/home/defne-odabasi/PycharmProjects/yolov1_project/archive/labels'
dog_csv_path = 'dogs.csv'
person_csv_path = 'persons.csv'
dog_person_csv_path = 'dog_person.csv'

# Initialize lists to store file names
dog_files = []
person_files = []
dog_person_files = []

# Process each label file
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()

        classes_present = set(int(line.split()[0]) for line in lines)

        img_name = label_file.replace('.txt', '.jpg')
        label_name = label_file

        if classes_present == {11}:
            dog_files.append((img_name, label_name))
        elif classes_present == {14}:
            person_files.append((img_name, label_name))
        elif classes_present == {11, 14}:
            dog_person_files.append((img_name, label_name))

# Convert lists to DataFrames and save as CSV
dog_df = pd.DataFrame(dog_files, columns=['img', 'label'])
person_df = pd.DataFrame(person_files, columns=['img', 'label'])
dog_person_df = pd.DataFrame(dog_person_files, columns=['img', 'label'])

dog_df.to_csv(dog_csv_path, index=False)
person_df.to_csv(person_csv_path, index=False)
dog_person_df.to_csv(dog_person_csv_path, index=False)

print(f'Successfully created {dog_csv_path}, {person_csv_path}, and {dog_person_csv_path}.')


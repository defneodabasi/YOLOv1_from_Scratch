import pandas as pd
import os
from sklearn.model_selection import train_test_split

# File paths
train_csv = 'archive/train.csv'
test_csv = 'archive/test.csv'

# Load CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Split the training data into training and validation sets
train_set_df, val_set_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Save to CSV files
train_set_df.to_csv('train_set.csv', index=False)
val_set_df.to_csv('val_set.csv', index=False)

# Save to CSV files
train_set_df.to_csv('train_set.csv', index=False)
val_set_df.to_csv('val_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)

# Verify the split
print(f"Number of lines in train_set.csv: {len(train_set_df)}")
print(f"Number of lines in val_set.csv: {len(val_set_df)}")
print(f"Number of lines in test_set.csv: {len(test_df)}")

import pandas as pd

# Load
segments = pd.read_csv('path/to/segments.csv')   #TODO: replace with actual path
dataset = pd.read_csv('path/to/dataset.csv')

# Merge
merged = segments.merge(dataset, on=['segment', 'channel'], how='left', suffixes=('', '_feat'))

# Remove duplicate columns if same info
# anomaly and train exist in both; keep original in segments
merged = merged.drop(columns=['anomaly_feat', 'train_feat'])

# Save
output_path = 'merged_data.csv'
merged.to_csv(output_path, index=False)

output_path

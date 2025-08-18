import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (ensure the CSV file is in the same directory or adjust the path)
df = pd.read_csv("CSVs\OPSAT-AD_modified.csv")

# Define key features to visualize
key_features = [
    'mean', 'var', 'kurtosis', 'skew',
    'n_peaks', 'diff_peaks', 'diff2_peaks', 'gaps_squared', 'var_div_duration'
]

# Create histograms for each feature split by anomaly label
plt.figure(figsize=(20, 20))
for i, feature in enumerate(key_features):
    plt.subplot(5, 2, i + 1)
    for label in sorted(df['anomaly'].unique()):
        subset = df[df['anomaly'] == label]
        plt.hist(subset[feature], bins=50, alpha=0.5, label=f"Anomaly={label}", density=True)
    plt.title(f'Distribution of {feature} by Anomaly Label')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()


plt.tight_layout()
plt.show()

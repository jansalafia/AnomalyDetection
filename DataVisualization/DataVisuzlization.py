import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv("CSVs\OPSAT-AD_modified.csv")


# Create output directory
import os
os.makedirs("figures", exist_ok=True)

# 1. Anomaly Label Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='anomaly', data=df)
plt.title('Anomaly Label Distribution')
plt.xlabel('Anomaly')
plt.ylabel('Count')
plt.xticks([0, 1], ['Normal', 'Anomalous'])
plt.tight_layout()
plt.savefig("figures/anomaly_distribution.png")
plt.close()

# 2. Standard Deviation Distribution by Anomaly
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='std', hue='anomaly', kde=True, bins=30)
plt.title('Distribution of Standard Deviation by Anomaly')
plt.xlabel('Standard Deviation')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("figures/std_distribution_by_anomaly.png")
plt.close()

# 4. Principal Component Analysis Scatter Plot
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
features = df[numeric_cols].drop(columns=['anomaly', 'train', 'sampling', 'segment'], errors='ignore')
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

# Add PCA to DataFrame
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='anomaly', palette='Set1', alpha=0.7)
plt.title('PCA of Telemetry Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.savefig("figures/pca_scatter.png")
plt.close()

print("Visualizations saved to 'figures' directory.")

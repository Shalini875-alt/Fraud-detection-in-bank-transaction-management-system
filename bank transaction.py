import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("bank_transactions_data.csv")

# Data Overview
print("Dataset Shape:", df.shape)
print("\nColumn Data Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nFirst Few Rows:\n", df.head())
print("\nDescriptive Statistics:\n", df.describe())

# Preprocessing: Ensure proper data formats
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
df['CustomerAge'] = pd.to_numeric(df['CustomerAge'], errors='coerce')
df = df.dropna(subset=['TransactionAmount', 'CustomerAge'])  # Drop rows with missing values

# Select Features for Analysis
X = df[['TransactionAmount', 'CustomerAge']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means Clustering
print("\n--- Applying K-means Clustering ---")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

# Calculate distances to cluster centroids and flag frauds
distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[kmeans_labels], axis=1)
threshold_kmeans = np.percentile(distances, 95)  # Top 5% as potential frauds
df['Potential_Fraud_KMeans'] = distances > threshold_kmeans

# Isolation Forest for Anomaly Detection
print("\n--- Applying Isolation Forest ---")
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
isolation_labels = isolation_forest.fit_predict(X_scaled)
df['IsolationForest_Score'] = isolation_labels
df['Potential_Fraud_IsolationForest'] = isolation_labels == -1  # Flag anomalies as frauds

# DBSCAN for Density-Based Clustering
print("\n--- Applying DBSCAN Clustering ---")
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = dbscan_labels
df['Potential_Fraud_DBSCAN'] = dbscan_labels == -1  # Outliers (-1) are potential frauds

# Visualizations
print("\n--- Visualizing Clustering and Fraud Detection ---")

# K-means Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans_labels, palette='viridis', s=60, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Scaled Transaction Amount')
plt.ylabel('Scaled Customer Age')
plt.legend()

# Isolation Forest Visualization
plt.subplot(1, 3, 2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Potential_Fraud_IsolationForest'],
                palette={False: 'blue', True: 'red'}, s=60, alpha=0.7)
plt.title('Isolation Forest Anomalies')
plt.xlabel('Scaled Transaction Amount')
plt.ylabel('Scaled Customer Age')

# DBSCAN Visualization
plt.subplot(1, 3, 3)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=dbscan_labels, palette='Set1', s=60, alpha=0.7)
plt.title('DBSCAN Clustering')
plt.xlabel('Scaled Transaction Amount')
plt.ylabel('Scaled Customer Age')

plt.tight_layout()
plt.show()

# Summary of Fraud Detection
print("\n--- Fraud Detection Summary ---")
print(f"K-means Detected Frauds: {df['Potential_Fraud_KMeans'].sum()}")
print(f"Isolation Forest Detected Frauds: {df['Potential_Fraud_IsolationForest'].sum()}")
print(f"DBSCAN Detected Frauds: {df['Potential_Fraud_DBSCAN'].sum()}")

# Save Results
output_file = "fraud_detection_results.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to '{output_file}'.")

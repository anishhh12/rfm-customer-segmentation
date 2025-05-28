
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D
import os

def load_and_preprocess(filepath):
    df = pd.read_excel(filepath)
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df[df['Country'] == 'United Kingdom']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    return df

def calculate_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.DateOffset(1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Revenue': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]
    return rfm

def scale_rfm(rfm):
    scaler = StandardScaler()
    return scaler.fit_transform(rfm)

def optimal_kmeans_clusters(rfm_scaled, cluster_range=range(2, 11)):
    scores = []
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        score = davies_bouldin_score(rfm_scaled, kmeans.labels_)
        scores.append(score)
    optimal_k = cluster_range[np.argmin(scores)]
    return optimal_k

def plot_rfm_distributions(rfm, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(rfm['Recency'], bins=30, ax=axes[0], color='skyblue')
    sns.histplot(rfm['Frequency'], bins=30, ax=axes[1], color='green')
    sns.histplot(rfm['Monetary'], bins=30, ax=axes[2], color='red')
    axes[0].set_title("Recency")
    axes[1].set_title("Frequency")
    axes[2].set_title("Monetary")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rfm_distributions.png"))
    plt.close()

def main():
    os.makedirs("images", exist_ok=True)
    df = load_and_preprocess("data/Online Retail.xlsx")
    rfm = calculate_rfm(df)
    rfm_scaled = scale_rfm(rfm)

    # Plot distributions
    plot_rfm_distributions(rfm, "images")

    # Optimal K
    k = optimal_kmeans_clusters(rfm_scaled)
    print(f"Optimal Clusters (KMeans): {k}")

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["Cluster_KMeans"] = kmeans.fit_predict(rfm_scaled)

    # Save segmented data
    rfm.to_csv("rfm_segmented.csv")
    print("Segmented RFM data saved to rfm_segmented.csv")

if __name__ == "__main__":
    main()

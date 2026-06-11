from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from preprocessing import build_preprocessor, prepare_features
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import scipy.cluster.hierarchy as sch


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "marketing_campaign.csv"

# 1. Load the dataset
df = pd.read_csv(DATA_PATH, sep="\t")
features = prepare_features(df)
preprocessor = build_preprocessor(features)

# 2. Fit the preprocessor to the data
X = preprocessor.fit_transform(features)

# 3. Elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

results = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(preprocessor.transform(features))
    results.append(kmeans.inertia_)

# 4. Plot the elbow curve
# plt.plot(range(1, 11), results)
# plt.xlabel("Number of Clusters")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.show()



# 5.1 Fit KMeans with the optimal number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df['KMeans_Cluster'] = clusters

# 5.2 Fit Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_clusters = agglo.fit_predict(X)
df['Agglo_Cluster'] = agglo_clusters


# Dendogram for Agglomerative Clustering
# plt.figure(figsize=(10,5))
# sch.dendrogram(sch.linkage(X, method='ward'))
# plt.title("Dendrogram")
# plt.show()


# PCA for visualization (Before DBSCAN Outliers Removal)
pca = PCA(n_components=3)
X_2d = pca.fit_transform(X)

plt.scatter(X_2d[:,0], X_2d[:,1], c=clusters)
plt.title("KMeans Clusters (PCA)")
plt.show()

def get_metrics(name, X, labels):
    return {
        "Model": name,
        "Silhouette Score": silhouette_score(X, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(X, labels),
        "Davies-Bouldin Score": davies_bouldin_score(X, labels)
    }


results = []

results.append(get_metrics("KMeans", X, clusters))
results.append(get_metrics("Agglomerative", X, agglo_clusters))

metrics_df = pd.DataFrame(results)
print(metrics_df)
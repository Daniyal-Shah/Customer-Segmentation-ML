from pathlib import Path

import pandas as pd
from preprocessing import build_preprocessor, prepare_features

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
plt.plot(range(1, 11), results)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
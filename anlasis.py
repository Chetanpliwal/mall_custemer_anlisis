import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# 2. Select features for segmentation (Annual Income and Spending Score)
X = df.iloc[:, [3, 4]].values

# 3. Find optimal clusters using the Elbow Method (Optional check)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 4. Apply K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 5. Add cluster labels back to dataframe
df['Cluster'] = y_kmeans

# Save segmented data to a new CSV for visualization tools
df.to_csv('segmented_customers.csv', index=False)
print("Clustering complete! File saved as segmented_customers.csv")
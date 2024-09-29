import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import itertools

# Load Iris dataset and create DataFrame
iris = load_iris()
df_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

# Separate features and target
X = df_data.drop(labels=['Species'], axis=1).values
y = df_data['Species']

# Check for missing values
print("Checked missing data (NAN count):", len(np.where(np.isnan(X))[0]))

# K-Means clustering
kmeansModel = KMeans(n_clusters=3, random_state=46)
clusters_pred = kmeansModel.fit_predict(X)

# Output model results
print("Inertia:", kmeansModel.inertia_)
print("Cluster Centers:\n", kmeansModel.cluster_centers_)

# True labels plot (based on species)
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", hue='Species', data=df_data, fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.show()

# K-Means predicted clusters
df_data['Predict'] = clusters_pred
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=df_data, hue="Predict", fit_reg=False, legend=False)
plt.scatter(kmeansModel.cluster_centers_[:, 2], kmeansModel.cluster_centers_[:, 3], s=200, c="r", marker='*')
plt.legend(title='Clusters', loc='upper left', labels=['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.title('K-Means Predicted Clusters with Centroids')
plt.show()

# Confusion Matrix to compare true labels vs predicted clusters
# cm = confusion_matrix(y, clusters_pred)
# # Function to plot confusion matrix as a heatmap
# def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None):
#     if cmap is None:
#         cmap = plt.get_cmap('Blues')
#     plt.figure(figsize=(6, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(target_names))
#     plt.xticks(tick_marks, target_names, rotation=45)
#     plt.yticks(tick_marks, target_names)
#
#     # Normalize the confusion matrix
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     thresh = cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, "{:0.2f}".format(cm[i, j]),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#
#
# # Plot confusion matrix
# plot_confusion_matrix(cm, target_names=['Setosa', 'Versicolour', 'Virginica'],
#                       title="Confusion Matrix: True vs Predicted Clusters")
# plt.show()
# 使用 inertia 做模型評估:使用inertia 迅速下降轉為平緩的那個點。
# K-means inertia for k = 1 to 9
kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_list]

# Elbow method plot
plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("Number of clusters (k)", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.title('Elbow Method for Optimal k')

# Add annotation for the elbow
plt.annotate('Elbow Point', xy=(3, inertias[2]), xytext=(4, inertias[2] + 300),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

# Set axis limits
plt.axis([1, 9, 0, 1300])

plt.show()
#  silhouette scores 做模型評估:分數越大越好
from sklearn.metrics import silhouette_score

# K-means clustering for k=1 to 9 (silhouette scores start from k=2)
kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(X) for k in range(2, 10)]

# Compute silhouette scores for k=2 to 9
silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_list]

# Plot silhouette scores
plt.figure(figsize=(8, 3.5))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("Number of clusters (k)", fontsize=14)
plt.ylabel("Silhouette Score", fontsize=14)
plt.title("Silhouette Scores for Different k Values")

# Set y-axis limits for silhouette scores
plt.ylim(0, 1)

plt.show()


plt.show()
plt.show()


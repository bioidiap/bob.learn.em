import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from bob.learn.em import KMeansMachine

iris_data = load_iris()
data = iris_data.data
setosa = data[iris_data.target == 0]
versicolor = data[iris_data.target == 1]
virginica = data[iris_data.target == 2]

# Training KMeans
# 3 clusters with a feature dimensionality of 2
machine = KMeansMachine(n_clusters=3, init_method="k-means++").fit(data)

predictions = machine.predict(data)

# Plotting
figure, ax = plt.subplots()
plt.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
plt.scatter(
    versicolor[:, 0], versicolor[:, 1], c="goldenrod", label="versicolor"
)
plt.scatter(virginica[:, 0], virginica[:, 1], c="dimgrey", label="virginica")
plt.scatter(
    machine.centroids_[:, 0],
    machine.centroids_[:, 1],
    c="blue",
    marker="x",
    label="centroids",
    s=60,
)
plt.legend()
plt.xticks([], [])
plt.yticks([], [])
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal width")
plt.tight_layout()
plt.show()

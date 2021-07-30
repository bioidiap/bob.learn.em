from bob.learn.em.cluster import KMeansMachine
from bob.learn.em.cluster import KMeansTrainer
from bob.learn.em import train
import bob.db.iris
import numpy
import matplotlib.pyplot as plt

data_per_class = bob.db.iris.data()
setosa = numpy.column_stack(
    (data_per_class['setosa'][:, 0], data_per_class['setosa'][:, 3]))
versicolor = numpy.column_stack(
    (data_per_class['versicolor'][:, 0], data_per_class['versicolor'][:, 3]))
virginica = numpy.column_stack(
    (data_per_class['virginica'][:, 0], data_per_class['virginica'][:, 3]))

data = numpy.vstack((setosa, versicolor, virginica))

# Training KMeans
# 3 clusters with a feature dimensionality of 2
machine = KMeansMachine(3, 2)
trainer = KMeansTrainer(init_method="k-means++")
train(trainer, machine, data, max_iterations=200, convergence_threshold=1e-5)

# Plotting
figure, ax = plt.subplots()
plt.scatter(setosa[:, 0],
            setosa[:, 1], c="darkcyan", label="setosa")
plt.scatter(versicolor[:, 0],
            versicolor[:, 1], c="goldenrod", label="versicolor")
plt.scatter(virginica[:, 0],
            virginica[:, 1], c="dimgrey", label="virginica")
plt.scatter(machine.means[:, 0],
            machine.means[:, 1], c="blue", marker="x", label="centroids",
            s=60)
plt.legend()
plt.xticks([], [])
plt.yticks([], [])
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal width")
plt.tight_layout()
plt.savefig("iris_kmeans.png")

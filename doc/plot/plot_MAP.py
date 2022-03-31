import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import bob.learn.em
import numpy as np

np.random.seed(10)

iris_data = load_iris()
data = np.column_stack((iris_data.data[:, 0], iris_data.data[:, 3]))
setosa = data[iris_data.target == 0]
versicolor = data[iris_data.target == 1]
virginica = data[iris_data.target == 2]


# Two clusters with
mle_machine = bob.learn.em.GMMMachine(3)
# Creating some fake means for the example
mle_machine.means = np.array([[5, 3], [4, 2], [7, 3.0]])
mle_machine.variances = np.array([[0.1, 0.5], [0.2, 0.2], [0.7, 0.5]])


# Creating some random data centered in
map_machine = bob.learn.em.GMMMachine(
    3, trainer="map", ubm=mle_machine, map_relevance_factor=4
).fit(data)


figure, ax = plt.subplots()
# plt.scatter(data[:, 0], data[:, 1], c="olivedrab", label="new data")
plt.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
plt.scatter(
    versicolor[:, 0], versicolor[:, 1], c="goldenrod", label="versicolor"
)
plt.scatter(virginica[:, 0], virginica[:, 1], c="dimgrey", label="virginica")
plt.scatter(
    mle_machine.means[:, 0],
    mle_machine.means[:, 1],
    c="blue",
    marker="x",
    label="prior centroids - mle",
    s=60,
)
plt.scatter(
    map_machine.means[:, 0],
    map_machine.means[:, 1],
    c="red",
    marker="^",
    label="adapted centroids - map",
    s=60,
)
plt.legend()
plt.xticks([], [])
plt.yticks([], [])
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal width")
plt.tight_layout()
plt.show()

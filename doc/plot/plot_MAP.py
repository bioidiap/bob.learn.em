import matplotlib.pyplot as plt
import bob.db.iris
import bob.learn.em
import numpy
numpy.random.seed(10)

data_per_class = bob.db.iris.data()
setosa = numpy.column_stack(
    (data_per_class['setosa'][:, 0], data_per_class['setosa'][:, 3]))
versicolor = numpy.column_stack(
    (data_per_class['versicolor'][:, 0], data_per_class['versicolor'][:, 3]))
virginica = numpy.column_stack(
    (data_per_class['virginica'][:, 0], data_per_class['virginica'][:, 3]))

data = numpy.vstack((setosa, versicolor, virginica))

# Two clusters with a feature dimensionality of 3
mle_machine = bob.learn.em.GMMMachine(3, 2)
mle_machine.means = numpy.array([[5, 3], [4, 2], [7, 3.]])

# Creating some random data centered in
map_machine = bob.learn.em.GMMMachine(3, 2)
map_trainer = bob.learn.em.MAP_GMMTrainer(mle_machine, relevance_factor=4)
bob.learn.em.train(map_trainer, map_machine, data, max_iterations=200,
                   convergence_threshold=1e-5)  # Train the KMeansMachine


figure, ax = plt.subplots()
# plt.scatter(data[:, 0], data[:, 1], c="olivedrab", label="new data")
plt.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
plt.scatter(versicolor[:, 0], versicolor[:, 1],
            c="goldenrod", label="versicolor")
plt.scatter(virginica[:, 0], virginica[:, 1],
            c="dimgrey", label="virginica")
plt.scatter(mle_machine.means[:, 0],
            mle_machine.means[:, 1], c="blue", marker="x",
            label="prior centroids - mle", s=60)
plt.scatter(map_machine.means[:, 0], map_machine.means[:, 1], c="red",
            marker="^", label="adapted centroids - map", s=60)
plt.legend()
plt.xticks([], [])
plt.yticks([], [])
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal width")
plt.tight_layout()
plt.show()

import bob.learn.em
import bob.db.iris
import numpy
numpy.random.seed(10)
from matplotlib import pyplot

data_per_class = bob.db.iris.data()
setosa = numpy.column_stack((data_per_class['setosa'][:, 0], data_per_class['setosa'][:, 3]))
versicolor = numpy.column_stack((data_per_class['versicolor'][:, 0], data_per_class['versicolor'][:, 3]))
virginica = numpy.column_stack((data_per_class['virginica'][:, 0], data_per_class['virginica'][:, 3]))

data = numpy.vstack((setosa, versicolor, virginica))

mle_machine = bob.learn.em.GMMMachine(3, 2)# Two clusters with a feature dimensionality of 3
mle_trainer = bob.learn.em.ML_GMMTrainer(True, True, True)
mle_machine.means = numpy.array([[5, 3], [4, 2], [7, 3.]])
bob.learn.em.train(mle_trainer, mle_machine, data, max_iterations=200, convergence_threshold=1e-5) # Train the KMeansMachine

#Creating some random data centered in
new_data = numpy.random.normal(2, 0.8, (50, 2))
map_machine = bob.learn.em.GMMMachine(3, 2) # Two clusters with a feature dimensionality of 3
map_trainer = bob.learn.em.MAP_GMMTrainer(mle_machine, relevance_factor=4)
bob.learn.em.train(map_trainer, map_machine, new_data, max_iterations=200, convergence_threshold=1e-5) # Train the KMeansMachine


figure, ax = pyplot.subplots()
pyplot.scatter(new_data[:, 0], new_data[:, 1], c="olivedrab", label="new data")
pyplot.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
pyplot.scatter(versicolor[:, 0], versicolor[:, 1], c="goldenrod", label="versicolor")
pyplot.scatter(virginica[:, 0], virginica[:, 1], c="dimgrey", label="virginica")
pyplot.scatter(mle_machine.means[:, 0], mle_machine.means[:, 1], c="blue", marker="x", label="prior centroids - mle", s=60)
pyplot.scatter(map_machine.means[:, 0], map_machine.means[:, 1], c="red", marker="^", label="adapted centroids - map", s=60)
pyplot.legend()
ax.set_xticklabels("" for item in ax.get_xticklabels())
ax.set_yticklabels("" for item in ax.get_yticklabels())
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal width")
pyplot.show()


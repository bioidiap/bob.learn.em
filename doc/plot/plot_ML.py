from bob.learn.em.mixture import MLGMMTrainer
from bob.learn.em.mixture import GMMMachine
from bob.learn.em.mixture import Gaussian
import bob.db.iris
import numpy
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

import logging
logger = logging.getLogger("bob.learn.em")
logger.setLevel("DEBUG")

data_per_class = bob.db.iris.data()
setosa = numpy.column_stack(
    (data_per_class['setosa'][:, 0], data_per_class['setosa'][:, 3]))
versicolor = numpy.column_stack(
    (data_per_class['versicolor'][:, 0], data_per_class['versicolor'][:, 3]))
virginica = numpy.column_stack(
    (data_per_class['virginica'][:, 0], data_per_class['virginica'][:, 3]))

data = numpy.vstack((setosa, versicolor, virginica))

# Two clusters with a feature dimensionality of 3
machine = GMMMachine(3, convergence_threshold=1e-5)

init_means = numpy.array([[5, 3], [4, 2], [7, 3]], dtype=float)
gaussians = numpy.array([Gaussian(m) for m in init_means])
trainer = MLGMMTrainer(
    init_method=gaussians,
    update_means=True,
    update_variances=True,
    update_weights=True,
)

trainer.initialize(machine, data)
repeat = 8
for step in range(10):
    print(f"Step {step*repeat} through {step*repeat+repeat} out of {10*repeat}.")
    for r in range(repeat):
        print(f"Step {step*repeat+r}")
        machine = machine.fit_partial(data, trainer=trainer)

    figure, ax = plt.subplots()
    ax.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
    ax.scatter(versicolor[:, 0], versicolor[:, 1],
                c="goldenrod", label="versicolor")
    ax.scatter(virginica[:, 0], virginica[:, 1],
                c="dimgrey", label="virginica")
    ax.scatter(
        machine.gaussians_["mean"][:, 0],
        machine.gaussians_["mean"][:, 1],
        c="blue", marker="x", label="centroids", s=60
    )

    for g in machine.gaussians_:
        covariance = numpy.diag(g["variance"])
        radius = numpy.sqrt(5.991)
        eigvals, eigvecs = numpy.linalg.eig(covariance)
        axis = numpy.sqrt(eigvals) * radius
        slope = eigvecs[1][0] / eigvecs[1][1]
        angle = 180.0 * numpy.arctan(slope) / numpy.pi

        e1 = Ellipse(
            g["mean"], axis[0], axis[1],
            angle=angle, linewidth=1, fill=False, zorder=2
        )
        ax.add_patch(e1)

    plt.legend()
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Petal width")
    plt.tight_layout()
    plt.show()

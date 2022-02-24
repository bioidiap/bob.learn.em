import logging

import matplotlib.pyplot as plt
import numpy

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import bob.db.iris

from bob.learn.em.mixture import GMMMachine

logger = logging.getLogger("bob.learn.em")
logger.setLevel("DEBUG")

data_per_class = bob.db.iris.data()
setosa = numpy.column_stack(
    (data_per_class["setosa"][:, 0], data_per_class["setosa"][:, 3])
)
versicolor = numpy.column_stack(
    (data_per_class["versicolor"][:, 0], data_per_class["versicolor"][:, 3])
)
virginica = numpy.column_stack(
    (data_per_class["virginica"][:, 0], data_per_class["virginica"][:, 3])
)

data = numpy.vstack((setosa, versicolor, virginica))

# Two clusters with a feature dimensionality of 3
machine = GMMMachine(
    3,
    convergence_threshold=1e-5,
    update_means=True,
    update_variances=True,
    update_weights=True,
)

# Initialize the means with known values (optional, skips kmeans)
machine.means = numpy.array([[5, 3], [4, 2], [7, 3]], dtype=float)
machine = machine.fit(data)


# Plotting
figure, ax = plt.subplots()
ax.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
ax.scatter(
    versicolor[:, 0], versicolor[:, 1], c="goldenrod", label="versicolor"
)
ax.scatter(virginica[:, 0], virginica[:, 1], c="dimgrey", label="virginica")
ax.scatter(
    machine.means[:, 0],
    machine.means[:, 1],
    c="blue",
    marker="x",
    label="centroids",
    s=60,
)

# Draw ellipses for covariance
for mean, variance in zip(machine.means, machine.variances):
    eigvals, eigvecs = numpy.linalg.eig(numpy.diag(variance))
    axis = numpy.sqrt(eigvals) * numpy.sqrt(5.991)
    angle = 180.0 * numpy.arctan(eigvecs[1][0] / eigvecs[1][1]) / numpy.pi
    ax.add_patch(
        Ellipse(mean, *axis, angle=angle, linewidth=1, fill=False, zorder=2)
    )

# Plot details (legend, axis labels)
plt.legend(
    handles=ax.get_legend_handles_labels()[0]
    + [Line2D([0], [0], color="black", label="covariances")]
)
plt.xticks([], [])
plt.yticks([], [])
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal width")
plt.tight_layout()
plt.show()

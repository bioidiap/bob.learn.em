import logging

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.datasets import load_iris

from bob.learn.em import GMMMachine

logger = logging.getLogger("bob.learn.em")
logger.setLevel("DEBUG")

iris_data = load_iris()
data = np.column_stack((iris_data.data[:, 0], iris_data.data[:, 3]))
setosa = data[iris_data.target == 0]
versicolor = data[iris_data.target == 1]
virginica = data[iris_data.target == 2]

# Two clusters with a feature dimensionality of 3
machine = GMMMachine(
    3,
    convergence_threshold=1e-5,
    update_means=True,
    update_variances=True,
    update_weights=True,
)

# Initialize the means with known values (optional, skips kmeans)
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


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance
    """
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(
            Ellipse(position, nsig * width, nsig * height, angle, **kwargs)
        )


# Draw ellipses for covariance
w_factor = 0.2 / np.max(machine.weights)
for w, pos, covar in zip(machine.weights, machine.means, machine.variances):
    draw_ellipse(pos, covar, alpha=w * w_factor)

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

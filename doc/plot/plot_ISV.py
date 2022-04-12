import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris

import bob.learn.em

np.random.seed(2)  # FIXING A SEED


# GENERATING DATA
iris_data = load_iris()
X = np.column_stack((iris_data.data[:, 0], iris_data.data[:, 3]))
y = iris_data.target

setosa = X[iris_data.target == 0]
versicolor = X[iris_data.target == 1]
virginica = X[iris_data.target == 2]

n_gaussians = 3
r_U = 1


# TRAINING THE PRIOR
ubm = bob.learn.em.GMMMachine(n_gaussians)
# Initializing with old bob initialization
ubm.means = np.array(
    [
        [5.0048631, 0.26047739],
        [5.83509503, 1.40530362],
        [6.76257257, 1.98965356],
    ]
)
ubm.variances = np.array(
    [
        [0.11311728, 0.05183813],
        [0.11587106, 0.08492455],
        [0.20482993, 0.10438209],
    ]
)

ubm.weights = np.array([0.36, 0.36, 0.28])

isv_machine = bob.learn.em.ISVMachine(r_U, em_iterations=50, ubm=ubm)
isv_machine.U = np.array(
    [[-0.150035, -0.44441, -1.67812, 2.47621, -0.52885, 0.659141]]
).T

isv_machine = isv_machine.fit(X, y)

# Variability direction
u0 = isv_machine.U[0:2, 0] / np.linalg.norm(isv_machine.U[0:2, 0])
u1 = isv_machine.U[2:4, 0] / np.linalg.norm(isv_machine.U[2:4, 0])
u2 = isv_machine.U[4:6, 0] / np.linalg.norm(isv_machine.U[4:6, 0])

figure, ax = plt.subplots()
plt.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
plt.scatter(
    versicolor[:, 0], versicolor[:, 1], c="goldenrod", label="versicolor"
)
plt.scatter(virginica[:, 0], virginica[:, 1], c="dimgrey", label="virginica")

plt.scatter(
    ubm.means[:, 0],
    ubm.means[:, 1],
    c="blue",
    marker="x",
    label="centroids - mle",
)
# plt.scatter(ubm.means[:, 0], ubm.means[:, 1], c="blue",
#             marker=".", label="within class varibility", s=0.01)

ax.arrow(
    ubm.means[0, 0],
    ubm.means[0, 1],
    u0[0],
    u0[1],
    fc="k",
    ec="k",
    head_width=0.05,
    head_length=0.1,
)
ax.arrow(
    ubm.means[1, 0],
    ubm.means[1, 1],
    u1[0],
    u1[1],
    fc="k",
    ec="k",
    head_width=0.05,
    head_length=0.1,
)
ax.arrow(
    ubm.means[2, 0],
    ubm.means[2, 1],
    u2[0],
    u2[1],
    fc="k",
    ec="k",
    head_width=0.05,
    head_length=0.1,
)
plt.text(
    ubm.means[0, 0] + u0[0],
    ubm.means[0, 1] + u0[1] - 0.1,
    r"$\mathbf{U}_1$",
    fontsize=15,
)
plt.text(
    ubm.means[1, 0] + u1[0],
    ubm.means[1, 1] + u1[1] - 0.1,
    r"$\mathbf{U}_2$",
    fontsize=15,
)
plt.text(
    ubm.means[2, 0] + u2[0],
    ubm.means[2, 1] + u2[1] - 0.1,
    r"$\mathbf{U}_3$",
    fontsize=15,
)

plt.xticks([], [])
plt.yticks([], [])

# plt.grid(True)
plt.xlabel("Sepal length")
plt.ylabel("Petal width")
plt.legend()
plt.tight_layout()
plt.show()

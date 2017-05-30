import bob.db.iris
import bob.learn.em
import bob.learn.linear
import matplotlib.pyplot as plt
import numpy
numpy.random.seed(2)  # FIXING A SEED


def train_ubm(features, n_gaussians):
    """
    Train UBM

     **Parameters**
       features: 2D numpy array with the features

       n_gaussians: Number of Gaussians

    """
    input_size = features.shape[1]

    kmeans_machine = bob.learn.em.KMeansMachine(int(n_gaussians), input_size)
    ubm = bob.learn.em.GMMMachine(int(n_gaussians), input_size)

    # The K-means clustering is firstly used to used to estimate the initial
    # means, the final variances and the final weights for each gaussian
    # component
    kmeans_trainer = bob.learn.em.KMeansTrainer('RANDOM_NO_DUPLICATE')
    bob.learn.em.train(kmeans_trainer, kmeans_machine, features)

    # Getting the means, weights and the variances for each cluster. This is a
    # very good estimator for the ML
    (variances, weights) = kmeans_machine.get_variances_and_weights_for_each_cluster(features)
    means = kmeans_machine.means

    # initialize the UBM with the output of kmeans
    ubm.means = means
    ubm.variances = variances
    ubm.weights = weights

    # Creating the ML Trainer. We will adapt only the means
    trainer = bob.learn.em.ML_GMMTrainer(
        update_means=True, update_variances=False, update_weights=False)
    bob.learn.em.train(trainer, ubm, features)

    return ubm


def isv_train(features, ubm):
    """
    Train U matrix

    **Parameters**
      features: List of :py:class:`bob.learn.em.GMMStats` organized by class

      n_gaussians: UBM (:py:class:`bob.learn.em.GMMMachine`)

    """

    stats = []
    for user in features:
        user_stats = []
        for f in user:
            s = bob.learn.em.GMMStats(ubm.shape[0], ubm.shape[1])
            ubm.acc_statistics(f, s)
            user_stats.append(s)
        stats.append(user_stats)

    relevance_factor = 4
    subspace_dimension_of_u = 1

    isvbase = bob.learn.em.ISVBase(ubm, subspace_dimension_of_u)
    trainer = bob.learn.em.ISVTrainer(relevance_factor)
    # trainer.rng = bob.core.random.mt19937(int(self.init_seed))
    bob.learn.em.train(trainer, isvbase, stats, max_iterations=50)

    return isvbase


# GENERATING DATA
data_per_class = bob.db.iris.data()
setosa = numpy.column_stack(
    (data_per_class['setosa'][:, 0], data_per_class['setosa'][:, 3]))
versicolor = numpy.column_stack(
    (data_per_class['versicolor'][:, 0], data_per_class['versicolor'][:, 3]))
virginica = numpy.column_stack(
    (data_per_class['virginica'][:, 0], data_per_class['virginica'][:, 3]))
data = numpy.vstack((setosa, versicolor, virginica))

# TRAINING THE PRIOR
ubm = train_ubm(data, 3)
isvbase = isv_train([setosa, versicolor, virginica], ubm)

# Variability direction
u0 = isvbase.u[0:2, 0] / numpy.linalg.norm(isvbase.u[0:2, 0])
u1 = isvbase.u[2:4, 0] / numpy.linalg.norm(isvbase.u[2:4, 0])
u2 = isvbase.u[4:6, 0] / numpy.linalg.norm(isvbase.u[4:6, 0])

figure, ax = plt.subplots()
plt.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
plt.scatter(versicolor[:, 0], versicolor[:, 1],
            c="goldenrod", label="versicolor")
plt.scatter(virginica[:, 0], virginica[:, 1], c="dimgrey", label="virginica")

plt.scatter(ubm.means[:, 0], ubm.means[:, 1], c="blue",
            marker="x", label="centroids - mle")
# plt.scatter(ubm.means[:, 0], ubm.means[:, 1], c="blue",
#             marker=".", label="within class varibility", s=0.01)

ax.arrow(ubm.means[0, 0], ubm.means[0, 1], u0[0], u0[1],
         fc="k", ec="k", head_width=0.05, head_length=0.1)
ax.arrow(ubm.means[1, 0], ubm.means[1, 1], u1[0], u1[1],
         fc="k", ec="k", head_width=0.05, head_length=0.1)
ax.arrow(ubm.means[2, 0], ubm.means[2, 1], u2[0], u2[1],
         fc="k", ec="k", head_width=0.05, head_length=0.1)
plt.text(ubm.means[0, 0] + u0[0], ubm.means[0, 1] +
         u0[1] - 0.1, r'$\mathbf{U}_1$', fontsize=15)
plt.text(ubm.means[1, 0] + u1[0], ubm.means[1, 1] +
         u1[1] - 0.1, r'$\mathbf{U}_2$', fontsize=15)
plt.text(ubm.means[2, 0] + u2[0], ubm.means[2, 1] +
         u2[1] - 0.1, r'$\mathbf{U}_3$', fontsize=15)

plt.xticks([], [])
plt.yticks([], [])

# plt.grid(True)
plt.xlabel('Sepal length')
plt.ylabel('Petal width')
plt.legend()
plt.tight_layout()
plt.show()

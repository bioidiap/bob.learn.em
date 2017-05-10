import bob.db.iris
import numpy

numpy.random.seed(2)  # FIXING A SEED
import bob.learn.linear
import bob.learn.em

import matplotlib.pyplot as plt


def train_ubm(features, n_gaussians):
    input_size = features.shape[1]

    kmeans_machine = bob.learn.em.KMeansMachine(int(n_gaussians), input_size)
    ubm = bob.learn.em.GMMMachine(int(n_gaussians), input_size)

    # The K-means clustering is firstly used to used to estimate the initial means, the final variances and the final weights for each gaussian component
    kmeans_trainer = bob.learn.em.KMeansTrainer('RANDOM_NO_DUPLICATE')
    bob.learn.em.train(kmeans_trainer, kmeans_machine, features)

    # Getting the means, weights and the variances for each cluster. This is a very good estimator for the ML
    (variances, weights) = kmeans_machine.get_variances_and_weights_for_each_cluster(features)
    means = kmeans_machine.means

    # initialize the UBM with the output of kmeans
    ubm.means = means
    ubm.variances = variances
    ubm.weights = weights

    # Creating the ML Trainer. We will adapt only the means
    trainer = bob.learn.em.ML_GMMTrainer(update_means=True, update_variances=False, update_weights=False)
    bob.learn.em.train(trainer, ubm, features)

    return ubm


def jfa_train(features, ubm):
    """
    Features com lista de listas [  [data_point_1_user_1,data_point_2_user_1], [data_point_1_user_2,data_point_2_user_2]  ] 
    """

    stats = []
    for user in features:
        user_stats = []
        for f in user:
            s = bob.learn.em.GMMStats(ubm.shape[0], ubm.shape[1])
            ubm.acc_statistics(f, s)
            user_stats.append(s)
        stats.append(user_stats)

    subspace_dimension_of_u = 1
    subspace_dimension_of_v = 1

    jfa_base = bob.learn.em.JFABase(ubm, subspace_dimension_of_u, subspace_dimension_of_v)
    trainer = bob.learn.em.JFATrainer()
    # trainer.rng = bob.core.random.mt19937(int(self.init_seed))
    bob.learn.em.train_jfa(trainer, jfa_base, stats, max_iterations=50)

    return jfa_base


### GENERATING DATA
data_per_class = bob.db.iris.data()
setosa = numpy.column_stack((data_per_class['setosa'][:, 0], data_per_class['setosa'][:, 3]))
versicolor = numpy.column_stack((data_per_class['versicolor'][:, 0], data_per_class['versicolor'][:, 3]))
virginica = numpy.column_stack((data_per_class['virginica'][:, 0], data_per_class['virginica'][:, 3]))
data = numpy.vstack((setosa, versicolor, virginica))

# TRAINING THE PRIOR
ubm = train_ubm(data, 3)
jfa_base = jfa_train([setosa, versicolor, virginica], ubm)

# Variability direction U
u0 = jfa_base.u[0:2, 0] / numpy.linalg.norm(jfa_base.u[0:2, 0])
u1 = jfa_base.u[2:4, 0] / numpy.linalg.norm(jfa_base.u[2:4, 0])
u2 = jfa_base.u[4:6, 0] / numpy.linalg.norm(jfa_base.u[4:6, 0])


# Variability direction V
v0 = jfa_base.v[0:2, 0] / numpy.linalg.norm(jfa_base.v[0:2, 0])
v1 = jfa_base.v[2:4, 0] / numpy.linalg.norm(jfa_base.v[2:4, 0])
v2 = jfa_base.v[4:6, 0] / numpy.linalg.norm(jfa_base.v[4:6, 0])



figure, ax = plt.subplots()
plt.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
plt.scatter(versicolor[:, 0], versicolor[:, 1], c="goldenrod", label="versicolor")
plt.scatter(virginica[:, 0], virginica[:, 1], c="dimgrey", label="virginica")

plt.scatter(ubm.means[:, 0], ubm.means[:, 1], c="blue", marker="x", label="centroids - mle")
#plt.scatter(ubm.means[:, 0], ubm.means[:, 1], c="blue", marker=".", label="within class varibility", s=0.01)

# U
ax.arrow(ubm.means[0, 0], ubm.means[0, 1], u0[0], u0[1], fc="k", ec="k", head_width=0.05, head_length=0.1)
ax.arrow(ubm.means[1, 0], ubm.means[1, 1], u1[0], u1[1], fc="k", ec="k", head_width=0.05, head_length=0.1)
ax.arrow(ubm.means[2, 0], ubm.means[2, 1], u2[0], u2[1], fc="k", ec="k", head_width=0.05, head_length=0.1)
plt.text(ubm.means[0, 0] + u0[0], ubm.means[0, 1] + u0[1] - 0.1, r'$\mathbf{U}_1$', fontsize=15)
plt.text(ubm.means[1, 0] + u1[0], ubm.means[1, 1] + u1[1] - 0.1, r'$\mathbf{U}_2$', fontsize=15)
plt.text(ubm.means[2, 0] + u2[0], ubm.means[2, 1] + u2[1] - 0.1, r'$\mathbf{U}_3$', fontsize=15)

# V
ax.arrow(ubm.means[0, 0], ubm.means[0, 1], v0[0], v0[1], fc="k", ec="k", head_width=0.05, head_length=0.1)
ax.arrow(ubm.means[1, 0], ubm.means[1, 1], v1[0], v1[1], fc="k", ec="k", head_width=0.05, head_length=0.1)
ax.arrow(ubm.means[2, 0], ubm.means[2, 1], v2[0], v2[1], fc="k", ec="k", head_width=0.05, head_length=0.1)
plt.text(ubm.means[0, 0] + v0[0], ubm.means[0, 1] + v0[1] - 0.1, r'$\mathbf{V}_1$', fontsize=15)
plt.text(ubm.means[1, 0] + v1[0], ubm.means[1, 1] + v1[1] - 0.1, r'$\mathbf{V}_2$', fontsize=15)
plt.text(ubm.means[2, 0] + v2[0], ubm.means[2, 1] + v2[1] - 0.1, r'$\mathbf{V}_3$', fontsize=15)

ax.set_xticklabels("" for item in ax.get_xticklabels())
ax.set_yticklabels("" for item in ax.get_yticklabels())

# plt.grid(True)
plt.xlabel('Sepal length')
plt.ylabel('Petal width')
plt.legend(loc=2)
plt.ylim([-1, 3.5])

#plt.show()

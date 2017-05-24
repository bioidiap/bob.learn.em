import bob.db.iris
import bob.learn.em
import bob.learn.linear
import matplotlib.pyplot as plt
import numpy
numpy.random.seed(2)  # FIXING A SEED


def train_ubm(features, n_gaussians):
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


def ivector_train(features, ubm):
    """
    Features com lista de listas [  [data_point_1_user_1,data_point_2_user_1],
    [data_point_1_user_2,data_point_2_user_2]  ]
    """

    stats = []
    for user in features:
        s = bob.learn.em.GMMStats(ubm.shape[0], ubm.shape[1])
        for f in user:
            ubm.acc_statistics(f, s)
        stats.append(s)

    subspace_dimension_of_t = 2

    ivector_trainer = bob.learn.em.IVectorTrainer(update_sigma=True)
    ivector_machine = bob.learn.em.IVectorMachine(
        ubm, subspace_dimension_of_t, 10e-5)

    # train IVector model
    bob.learn.em.train(ivector_trainer, ivector_machine, stats, 500)

    return ivector_machine


def acc_stats(data, gmm):
    gmm_stats = []
    for d in data:
        s = bob.learn.em.GMMStats(gmm.shape[0], gmm.shape[1])
        gmm.acc_statistics(d, s)
        gmm_stats.append(s)

    return gmm_stats


def compute_ivectors(gmm_stats, ivector_machine):

    ivectors = []
    for g in gmm_stats:
        ivectors.append(ivector_machine(g))

    return numpy.array(ivectors)


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
ivector_machine = ivector_train([setosa, versicolor, virginica], ubm)

# Variability direction U
# t0 = T[0:2, 0] / numpy.linalg.norm(T[0:2, 0])
# t1 = T[2:4, 0] / numpy.linalg.norm(T[2:4, 0])
# t2 = T[4:6, 0] / numpy.linalg.norm(T[4:6, 0])


# figure, ax = plt.subplots()
figure = plt.subplot(2, 1, 1)
ax = figure.axes
plt.title("Raw fetures")
plt.scatter(setosa[:, 0], setosa[:, 1], c="darkcyan", label="setosa")
plt.scatter(versicolor[:, 0], versicolor[:, 1],
            c="goldenrod", label="versicolor")
plt.scatter(virginica[:, 0], virginica[:, 1], c="dimgrey", label="virginica")
# plt.grid(True)
# plt.xlabel('Sepal length')
plt.ylabel('Petal width')
plt.legend(loc=2)
plt.ylim([-1, 3.5])
plt.xticks([], [])
plt.yticks([], [])


figure = plt.subplot(2, 1, 2)
ax = figure.axes
ivector_setosa = compute_ivectors(acc_stats(setosa, ubm), ivector_machine)
ivector_versicolor = compute_ivectors(
    acc_stats(versicolor, ubm), ivector_machine)
ivector_virginica = compute_ivectors(
    acc_stats(virginica, ubm), ivector_machine)


# Whitening iVectors
whitening_trainer = bob.learn.linear.WhiteningTrainer()
whitener_machine = bob.learn.linear.Machine(
    ivector_setosa.shape[1], ivector_setosa.shape[1])
whitening_trainer.train(numpy.vstack(
    (ivector_setosa, ivector_versicolor, ivector_virginica)), whitener_machine)
ivector_setosa = whitener_machine(ivector_setosa)
ivector_versicolor = whitener_machine(ivector_versicolor)
ivector_virginica = whitener_machine(ivector_virginica)


# LDA ivectors
lda_trainer = bob.learn.linear.FisherLDATrainer()
lda_machine = bob.learn.linear.Machine(
    ivector_setosa.shape[1], ivector_setosa.shape[1])
lda_trainer.train([ivector_setosa, ivector_versicolor,
                   ivector_virginica], lda_machine)
ivector_setosa = lda_machine(ivector_setosa)
ivector_versicolor = lda_machine(ivector_versicolor)
ivector_virginica = lda_machine(ivector_virginica)


# WCCN ivectors
# wccn_trainer = bob.learn.linear.WCCNTrainer()
# wccn_machine = bob.learn.linear.Machine(
#     ivector_setosa.shape[1], ivector_setosa.shape[1])
# wccn_trainer.train([ivector_setosa, ivector_versicolor,
#                     ivector_virginica], wccn_machine)
# ivector_setosa = wccn_machine(ivector_setosa)
# ivector_versicolor = wccn_machine(ivector_versicolor)
# ivector_virginica = wccn_machine(ivector_virginica)


plt.title("First two ivectors")
plt.scatter(ivector_setosa[:, 0],
            ivector_setosa[:, 1], c="darkcyan", label="setosa",
            marker="x")
plt.scatter(ivector_versicolor[:, 0],
            ivector_versicolor[:, 1], c="goldenrod", label="versicolor",
            marker="x")
plt.scatter(ivector_virginica[:, 0],
            ivector_virginica[:, 1], c="dimgrey", label="virginica",
            marker="x")

plt.xticks([], [])
plt.yticks([], [])

# plt.grid(True)
# plt.xlabel('Sepal length')
# plt.ylabel('Petal width')
plt.legend(loc=2)
plt.ylim([-1, 3.5])

plt.tight_layout()
plt.show()

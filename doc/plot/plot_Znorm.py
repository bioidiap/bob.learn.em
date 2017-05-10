import bob.learn.em
import numpy
numpy.random.seed(10)

from matplotlib import pyplot

n_clients = 10
n_scores_per_client = 200

# Defining some fake scores for genuines and impostors
impostor_scores = numpy.random.normal(-15.5, 5, (n_scores_per_client, n_clients))
genuine_scores = numpy.random.normal(0.5, 5, (n_scores_per_client, n_clients))

# Defining the scores for the statistics computation
z_scores = numpy.random.normal(-5., 5, (n_scores_per_client, n_clients))

# Z - Normalizing
z_norm_impostors = bob.learn.em.znorm(impostor_scores, z_scores)
z_norm_genuine = bob.learn.em.znorm(genuine_scores, z_scores)

# PLOTTING
figure = pyplot.subplot(2, 1, 1)
ax = figure.axes
pyplot.title("Raw scores", fontsize=8)
pyplot.hist(impostor_scores.reshape(n_scores_per_client*n_clients), label='Impostors', normed=True,
            color='red', alpha=0.5, bins=50)
pyplot.hist(genuine_scores.reshape(n_scores_per_client*n_clients), label='Genuine', normed=True,
            color='blue', alpha=0.5, bins=50)
pyplot.legend(fontsize=8)
ax.set_yticklabels("" for item in ax.get_yticklabels())


figure = pyplot.subplot(2, 1, 2)
ax = figure.axes
pyplot.title("Z-norm scores", fontsize=8)
pyplot.hist(z_norm_impostors.reshape(n_scores_per_client*n_clients), label='Z-Norm Impostors', normed=True,
            color='red', alpha=0.5, bins=50)
pyplot.hist(z_norm_genuine.reshape(n_scores_per_client*n_clients), label='Z-Norm Genuine', normed=True,
            color='blue', alpha=0.5, bins=50)
pyplot.legend(fontsize=8)
ax.set_yticklabels("" for item in ax.get_yticklabels())

pyplot.show()

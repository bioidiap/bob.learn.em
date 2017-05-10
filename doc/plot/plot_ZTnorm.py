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
t_scores = numpy.random.normal(-6., 5, (n_scores_per_client, n_clients))

# T-normalizing the Z-scores
zt_scores = bob.learn.em.tnorm(z_scores, t_scores)

# ZT - Normalizing
zt_norm_impostors = bob.learn.em.ztnorm(impostor_scores, z_scores, t_scores, zt_scores)
zt_norm_genuine = bob.learn.em.ztnorm(genuine_scores, z_scores, t_scores, zt_scores)

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
pyplot.title("T-norm scores", fontsize=8)
pyplot.hist(zt_norm_impostors.reshape(n_scores_per_client*n_clients), label='T-Norm Impostors', normed=True,
            color='red', alpha=0.5, bins=50)
pyplot.hist(zt_norm_genuine.reshape(n_scores_per_client*n_clients), label='T-Norm Genuine', normed=True,
            color='blue', alpha=0.5, bins=50)
pyplot.legend(fontsize=8)
ax.set_yticklabels("" for item in ax.get_yticklabels())

pyplot.show()

/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.learn.em/MAP_GMMTrainer.h>
#include <bob.core/check.h>

bob::learn::em::MAP_GMMTrainer::MAP_GMMTrainer(
   const bool update_means,
   const bool update_variances,
   const bool update_weights,
   const double mean_var_update_responsibilities_threshold,

   const bool reynolds_adaptation,
   const double relevance_factor,
   const double alpha,
   boost::shared_ptr<bob::learn::em::GMMMachine> prior_gmm):

  m_gmm_base_trainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold),
  m_prior_gmm(prior_gmm)
{
  m_reynolds_adaptation = reynolds_adaptation;
  m_relevance_factor    = relevance_factor;
  m_alpha               = alpha;
}


bob::learn::em::MAP_GMMTrainer::MAP_GMMTrainer(const bob::learn::em::MAP_GMMTrainer& b):
  m_gmm_base_trainer(b.m_gmm_base_trainer),
  m_prior_gmm(b.m_prior_gmm)
{
  m_relevance_factor    = b.m_relevance_factor;
  m_alpha               = b.m_alpha;
  m_reynolds_adaptation = b.m_reynolds_adaptation;
}

bob::learn::em::MAP_GMMTrainer::~MAP_GMMTrainer()
{}

void bob::learn::em::MAP_GMMTrainer::initialize(bob::learn::em::GMMMachine& gmm)
{
  // Check that the prior GMM has been specified
  if (!m_prior_gmm)
    throw std::runtime_error("MAP_GMMTrainer: Prior GMM distribution has not been set");

  // Allocate memory for the sufficient statistics and initialise
  m_gmm_base_trainer.initialize(gmm);

  const size_t n_gaussians = gmm.getNGaussians();
  // TODO: check size?
  gmm.setWeights(m_prior_gmm->getWeights());
  for(size_t i=0; i<n_gaussians; ++i)
  {
    gmm.getGaussian(i)->updateMean() = m_prior_gmm->getGaussian(i)->getMean();
    gmm.getGaussian(i)->updateVariance() = m_prior_gmm->getGaussian(i)->getVariance();
    gmm.getGaussian(i)->applyVarianceThresholds();
  }
  // Initializes cache
  m_cache_alpha.resize(n_gaussians);
  m_cache_ml_weights.resize(n_gaussians);
}

bool bob::learn::em::MAP_GMMTrainer::setPriorGMM(boost::shared_ptr<bob::learn::em::GMMMachine> prior_gmm)
{
  if (!prior_gmm) return false;
  m_prior_gmm = prior_gmm;
  return true;
}


void bob::learn::em::MAP_GMMTrainer::mStep(bob::learn::em::GMMMachine& gmm)
{
  // Read options and variables
  double n_gaussians = gmm.getNGaussians();

  //Checking if it is necessary to resize the cache
  if((size_t)m_cache_alpha.extent(0) != n_gaussians)
    initialize(gmm); //If it is different for some reason, there is no way, you have to initialize

  // Check that the prior GMM has been specified
  if (!m_prior_gmm)
    throw std::runtime_error("MAP_GMMTrainer: Prior GMM distribution has not been set");

  blitz::firstIndex i;
  blitz::secondIndex j;

  // Calculate the "data-dependent adaptation coefficient", alpha_i
  // TODO: check if required // m_cache_alpha.resize(n_gaussians);
  if (!m_reynolds_adaptation)
    m_cache_alpha = m_alpha;
  else
    m_cache_alpha = m_gmm_base_trainer.getGMMStats()->n(i) / (m_gmm_base_trainer.getGMMStats()->n(i) + m_relevance_factor);

  // - Update weights if requested
  //   Equation 11 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (m_gmm_base_trainer.getUpdateWeights()) {
    // Calculate the maximum likelihood weights
    m_cache_ml_weights = m_gmm_base_trainer.getGMMStats()->n / static_cast<double>(m_gmm_base_trainer.getGMMStats()->T); //cast req. for linux/32-bits & osx

    // Get the prior weights
    const blitz::Array<double,1>& prior_weights = m_prior_gmm->getWeights();
    blitz::Array<double,1>& new_weights = gmm.updateWeights();

    // Calculate the new weights
    new_weights = m_cache_alpha * m_cache_ml_weights + (1-m_cache_alpha) * prior_weights;

    // Apply the scale factor, gamma, to ensure the new weights sum to unity
    double gamma = blitz::sum(new_weights);
    new_weights /= gamma;

    // Recompute the log weights in the cache of the GMMMachine
    gmm.recomputeLogWeights();
  }

  // Update GMM parameters
  // - Update means if requested
  //   Equation 12 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (m_gmm_base_trainer.getUpdateMeans()) {
    // Calculate new means
    for (size_t i=0; i<n_gaussians; ++i) {
      const blitz::Array<double,1>& prior_means = m_prior_gmm->getGaussian(i)->getMean();
      blitz::Array<double,1>& means = gmm.getGaussian(i)->updateMean();
      if (m_gmm_base_trainer.getGMMStats()->n(i) < m_gmm_base_trainer.getMeanVarUpdateResponsibilitiesThreshold()) {
        means = prior_means;
      }
      else {
        // Use the maximum likelihood means
        means = m_cache_alpha(i) * (m_gmm_base_trainer.getGMMStats()->sumPx(i,blitz::Range::all()) / m_gmm_base_trainer.getGMMStats()->n(i)) + (1-m_cache_alpha(i)) * prior_means;
      }
    }
  }

  // - Update variance if requested
  //   Equation 13 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (m_gmm_base_trainer.getUpdateVariances()) {
    // Calculate new variances (equation 13)
    for (size_t i=0; i<n_gaussians; ++i) {
      const blitz::Array<double,1>& prior_means = m_prior_gmm->getGaussian(i)->getMean();
      blitz::Array<double,1>& means = gmm.getGaussian(i)->updateMean();
      const blitz::Array<double,1>& prior_variances = m_prior_gmm->getGaussian(i)->getVariance();
      blitz::Array<double,1>& variances = gmm.getGaussian(i)->updateVariance();
      if (m_gmm_base_trainer.getGMMStats()->n(i) < m_gmm_base_trainer.getMeanVarUpdateResponsibilitiesThreshold()) {
        variances = (prior_variances + prior_means) - blitz::pow2(means);
      }
      else {
        variances = m_cache_alpha(i) * m_gmm_base_trainer.getGMMStats()->sumPxx(i,blitz::Range::all()) / m_gmm_base_trainer.getGMMStats()->n(i) + (1-m_cache_alpha(i)) * (prior_variances + prior_means) - blitz::pow2(means);
      }
      gmm.getGaussian(i)->applyVarianceThresholds();
    }
  }
}



bob::learn::em::MAP_GMMTrainer& bob::learn::em::MAP_GMMTrainer::operator=
  (const bob::learn::em::MAP_GMMTrainer &other)
{
  if (this != &other)
  {
    m_gmm_base_trainer    = other.m_gmm_base_trainer;
    m_relevance_factor    = other.m_relevance_factor;
    m_prior_gmm           = other.m_prior_gmm;
    m_alpha               = other.m_alpha;
    m_reynolds_adaptation = other.m_reynolds_adaptation;
    m_cache_alpha.resize(other.m_cache_alpha.extent(0));
    m_cache_ml_weights.resize(other.m_cache_ml_weights.extent(0));
  }
  return *this;
}


bool bob::learn::em::MAP_GMMTrainer::operator==
  (const bob::learn::em::MAP_GMMTrainer &other) const
{
  return m_gmm_base_trainer    == other.m_gmm_base_trainer &&
         m_relevance_factor    == other.m_relevance_factor &&
         m_prior_gmm           == other.m_prior_gmm &&
         m_alpha               == other.m_alpha &&
         m_reynolds_adaptation == other.m_reynolds_adaptation;
}


bool bob::learn::em::MAP_GMMTrainer::operator!=
  (const bob::learn::em::MAP_GMMTrainer &other) const
{
  return !(this->operator==(other));
}


bool bob::learn::em::MAP_GMMTrainer::is_similar_to
  (const bob::learn::em::MAP_GMMTrainer &other, const double r_epsilon,
   const double a_epsilon) const
{
  return //m_gmm_base_trainer.is_similar_to(other.m_gmm_base_trainer, r_epsilon, a_epsilon) &&
         bob::core::isClose(m_relevance_factor, other.m_relevance_factor, r_epsilon, a_epsilon) &&
         m_prior_gmm == other.m_prior_gmm &&
         bob::core::isClose(m_alpha, other.m_alpha, r_epsilon, a_epsilon) &&
         m_reynolds_adaptation == other.m_reynolds_adaptation;
}

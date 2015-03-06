/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.learn.em/ML_GMMTrainer.h>
#include <algorithm>

bob::learn::em::ML_GMMTrainer::ML_GMMTrainer(
   const bool update_means,
   const bool update_variances,
   const bool update_weights,
   const double mean_var_update_responsibilities_threshold
):
  m_gmm_base_trainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold)
{}



bob::learn::em::ML_GMMTrainer::ML_GMMTrainer(const bob::learn::em::ML_GMMTrainer& b):
  m_gmm_base_trainer(b.m_gmm_base_trainer)
{}

bob::learn::em::ML_GMMTrainer::~ML_GMMTrainer()
{}

void bob::learn::em::ML_GMMTrainer::initialize(bob::learn::em::GMMMachine& gmm)
{
  m_gmm_base_trainer.initialize(gmm);

  // Allocate cache
  size_t n_gaussians = gmm.getNGaussians();
  m_cache_ss_n_thresholded.resize(n_gaussians);
}


void bob::learn::em::ML_GMMTrainer::mStep(bob::learn::em::GMMMachine& gmm)
{
  // Read options and variables
  const size_t n_gaussians = gmm.getNGaussians();

 //Checking if it is necessary to resize the cache
 if((size_t)m_cache_ss_n_thresholded.extent(0) != n_gaussians)
   initialize(gmm); //If it is different for some reason, there is no way, you have to initialize

  // - Update weights if requested
  //   Equation 9.26 of Bishop, "Pattern recognition and machine learning", 2006
  if (m_gmm_base_trainer.getUpdateWeights()) {
    blitz::Array<double,1>& weights = gmm.updateWeights();
    weights = m_gmm_base_trainer.getGMMStats()->n / static_cast<double>(m_gmm_base_trainer.getGMMStats()->T); //cast req. for linux/32-bits & osx
    // Recompute the log weights in the cache of the GMMMachine
    gmm.recomputeLogWeights();
  }

  // Generate a thresholded version of m_ss.n
  for(size_t i=0; i<n_gaussians; ++i)
    m_cache_ss_n_thresholded(i) = std::max(m_gmm_base_trainer.getGMMStats()->n(i), m_gmm_base_trainer.getMeanVarUpdateResponsibilitiesThreshold());

  // Update GMM parameters using the sufficient statistics (m_ss)
  // - Update means if requested
  //   Equation 9.24 of Bishop, "Pattern recognition and machine learning", 2006
  if (m_gmm_base_trainer.getUpdateMeans()) {
    for(size_t i=0; i<n_gaussians; ++i) {
      blitz::Array<double,1>& means = gmm.getGaussian(i)->updateMean();
      means = m_gmm_base_trainer.getGMMStats()->sumPx(i, blitz::Range::all()) / m_cache_ss_n_thresholded(i);
    }
  }

  // - Update variance if requested
  //   See Equation 9.25 of Bishop, "Pattern recognition and machine learning", 2006
  //   ...but we use the "computational formula for the variance", i.e.
  //   var = 1/n * sum (P(x-mean)(x-mean))
  //       = 1/n * sum (Pxx) - mean^2
  if (m_gmm_base_trainer.getUpdateVariances()) {
    for(size_t i=0; i<n_gaussians; ++i) {
      const blitz::Array<double,1>& means = gmm.getGaussian(i)->getMean();
      blitz::Array<double,1>& variances = gmm.getGaussian(i)->updateVariance();
      variances = m_gmm_base_trainer.getGMMStats()->sumPxx(i, blitz::Range::all()) / m_cache_ss_n_thresholded(i) - blitz::pow2(means);
      gmm.getGaussian(i)->applyVarianceThresholds();
    }
  }
}

bob::learn::em::ML_GMMTrainer& bob::learn::em::ML_GMMTrainer::operator=
  (const bob::learn::em::ML_GMMTrainer &other)
{
  if (this != &other)
  {
    m_gmm_base_trainer = other.m_gmm_base_trainer;
    m_cache_ss_n_thresholded.resize(other.m_cache_ss_n_thresholded.extent(0));
  }
  return *this;
}

bool bob::learn::em::ML_GMMTrainer::operator==
  (const bob::learn::em::ML_GMMTrainer &other) const
{
  return m_gmm_base_trainer == other.m_gmm_base_trainer;
}

bool bob::learn::em::ML_GMMTrainer::operator!=
  (const bob::learn::em::ML_GMMTrainer &other) const
{
  return !(this->operator==(other));
}

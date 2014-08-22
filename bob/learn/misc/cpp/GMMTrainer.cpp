/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.learn.misc/GMMTrainer.h>
#include <bob.core/assert.h>
#include <bob.core/check.h>

bob::learn::misc::GMMTrainer::GMMTrainer(const bool update_means,
    const bool update_variances, const bool update_weights,
    const double mean_var_update_responsibilities_threshold):
  bob::learn::misc::EMTrainer<bob::learn::misc::GMMMachine, blitz::Array<double,2> >(),
  m_update_means(update_means), m_update_variances(update_variances),
  m_update_weights(update_weights),
  m_mean_var_update_responsibilities_threshold(mean_var_update_responsibilities_threshold)
{
}

bob::learn::misc::GMMTrainer::GMMTrainer(const bob::learn::misc::GMMTrainer& b):
  bob::learn::misc::EMTrainer<bob::learn::misc::GMMMachine, blitz::Array<double,2> >(b),
  m_update_means(b.m_update_means), m_update_variances(b.m_update_variances),
  m_mean_var_update_responsibilities_threshold(b.m_mean_var_update_responsibilities_threshold)
{
}

bob::learn::misc::GMMTrainer::~GMMTrainer()
{
}

void bob::learn::misc::GMMTrainer::initialize(bob::learn::misc::GMMMachine& gmm,
  const blitz::Array<double,2>& data)
{
  // Allocate memory for the sufficient statistics and initialise
  m_ss.resize(gmm.getNGaussians(),gmm.getNInputs());
}

void bob::learn::misc::GMMTrainer::eStep(bob::learn::misc::GMMMachine& gmm,
  const blitz::Array<double,2>& data)
{
  m_ss.init();
  // Calculate the sufficient statistics and save in m_ss
  gmm.accStatistics(data, m_ss);
}

double bob::learn::misc::GMMTrainer::computeLikelihood(bob::learn::misc::GMMMachine& gmm)
{
  return m_ss.log_likelihood / m_ss.T;
}

void bob::learn::misc::GMMTrainer::finalize(bob::learn::misc::GMMMachine& gmm,
  const blitz::Array<double,2>& data)
{
}

bob::learn::misc::GMMTrainer& bob::learn::misc::GMMTrainer::operator=
  (const bob::learn::misc::GMMTrainer &other)
{
  if (this != &other)
  {
    bob::learn::misc::EMTrainer<bob::learn::misc::GMMMachine,
      blitz::Array<double,2> >::operator=(other);
    m_ss = other.m_ss;
    m_update_means = other.m_update_means;
    m_update_variances = other.m_update_variances;
    m_update_weights = other.m_update_weights;
    m_mean_var_update_responsibilities_threshold = other.m_mean_var_update_responsibilities_threshold;
  }
  return *this;
}

bool bob::learn::misc::GMMTrainer::operator==
  (const bob::learn::misc::GMMTrainer &other) const
{
  return bob::learn::misc::EMTrainer<bob::learn::misc::GMMMachine,
           blitz::Array<double,2> >::operator==(other) &&
         m_ss == other.m_ss &&
         m_update_means == other.m_update_means &&
         m_update_variances == other.m_update_variances &&
         m_update_weights == other.m_update_weights &&
         m_mean_var_update_responsibilities_threshold == other.m_mean_var_update_responsibilities_threshold;
}

bool bob::learn::misc::GMMTrainer::operator!=
  (const bob::learn::misc::GMMTrainer &other) const
{
  return !(this->operator==(other));
}

bool bob::learn::misc::GMMTrainer::is_similar_to
  (const bob::learn::misc::GMMTrainer &other, const double r_epsilon,
   const double a_epsilon) const
{
  return bob::learn::misc::EMTrainer<bob::learn::misc::GMMMachine,
           blitz::Array<double,2> >::operator==(other) &&
  // TODO: use is similar to method for the accumulator m_ss
         m_ss == other.m_ss &&
         m_update_means == other.m_update_means &&
         m_update_variances == other.m_update_variances &&
         m_update_weights == other.m_update_weights &&
         bob::core::isClose(m_mean_var_update_responsibilities_threshold,
          other.m_mean_var_update_responsibilities_threshold, r_epsilon, a_epsilon);
}

void bob::learn::misc::GMMTrainer::setGMMStats(const bob::learn::misc::GMMStats& stats)
{
  bob::core::array::assertSameShape(m_ss.sumPx, stats.sumPx);
  m_ss = stats;
}

/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.learn.em/GMMBaseTrainer.h>
#include <bob.core/assert.h>
#include <bob.core/check.h>

bob::learn::em::GMMBaseTrainer::GMMBaseTrainer(const bool update_means,
    const bool update_variances, const bool update_weights,
    const double mean_var_update_responsibilities_threshold):
  m_ss(new bob::learn::em::GMMStats()),
  m_update_means(update_means), m_update_variances(update_variances),
  m_update_weights(update_weights),
  m_mean_var_update_responsibilities_threshold(mean_var_update_responsibilities_threshold)
{}

bob::learn::em::GMMBaseTrainer::GMMBaseTrainer(const bob::learn::em::GMMBaseTrainer& b):
  m_ss(new bob::learn::em::GMMStats()),
  m_update_means(b.m_update_means), m_update_variances(b.m_update_variances),
  m_mean_var_update_responsibilities_threshold(b.m_mean_var_update_responsibilities_threshold)
{}

bob::learn::em::GMMBaseTrainer::~GMMBaseTrainer()
{}

void bob::learn::em::GMMBaseTrainer::initialize(bob::learn::em::GMMMachine& gmm)
{
  // Allocate memory for the sufficient statistics and initialise
  m_ss->resize(gmm.getNGaussians(),gmm.getNInputs());
}

void bob::learn::em::GMMBaseTrainer::eStep(bob::learn::em::GMMMachine& gmm,
  const blitz::Array<double,2>& data)
{
  m_ss->init();
  // Calculate the sufficient statistics and save in m_ss
  gmm.accStatistics(data, *m_ss);
}

double bob::learn::em::GMMBaseTrainer::computeLikelihood(bob::learn::em::GMMMachine& gmm)
{
  return m_ss->log_likelihood / m_ss->T;
}


bob::learn::em::GMMBaseTrainer& bob::learn::em::GMMBaseTrainer::operator=
  (const bob::learn::em::GMMBaseTrainer &other)
{
  if (this != &other)
  {
    *m_ss = *other.m_ss;
    m_update_means = other.m_update_means;
    m_update_variances = other.m_update_variances;
    m_update_weights = other.m_update_weights;
    m_mean_var_update_responsibilities_threshold = other.m_mean_var_update_responsibilities_threshold;
  }
  return *this;
}

bool bob::learn::em::GMMBaseTrainer::operator==
  (const bob::learn::em::GMMBaseTrainer &other) const
{
  return *m_ss == *other.m_ss &&
         m_update_means == other.m_update_means &&
         m_update_variances == other.m_update_variances &&
         m_update_weights == other.m_update_weights &&
         m_mean_var_update_responsibilities_threshold == other.m_mean_var_update_responsibilities_threshold;
}

bool bob::learn::em::GMMBaseTrainer::operator!=
  (const bob::learn::em::GMMBaseTrainer &other) const
{
  return !(this->operator==(other));
}

bool bob::learn::em::GMMBaseTrainer::is_similar_to
  (const bob::learn::em::GMMBaseTrainer &other, const double r_epsilon,
   const double a_epsilon) const
{
  return *m_ss == *other.m_ss &&
         m_update_means == other.m_update_means &&
         m_update_variances == other.m_update_variances &&
         m_update_weights == other.m_update_weights &&
         bob::core::isClose(m_mean_var_update_responsibilities_threshold,
          other.m_mean_var_update_responsibilities_threshold, r_epsilon, a_epsilon);
}

void bob::learn::em::GMMBaseTrainer::setGMMStats(boost::shared_ptr<bob::learn::em::GMMStats> stats)
{
  bob::core::array::assertSameShape(m_ss->sumPx, stats->sumPx);
  m_ss = stats;
}

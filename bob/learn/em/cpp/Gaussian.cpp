/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.learn.em/Gaussian.h>

#include <bob.core/assert.h>
#include <bob.math/log.h>

bob::learn::em::Gaussian::Gaussian() {
  resize(0);
}

bob::learn::em::Gaussian::Gaussian(const size_t n_inputs) {
  resize(n_inputs);
}

bob::learn::em::Gaussian::Gaussian(const bob::learn::em::Gaussian& other) {
  copy(other);
}

bob::learn::em::Gaussian::Gaussian(bob::io::base::HDF5File& config) {
  load(config);
}

bob::learn::em::Gaussian::~Gaussian() {
}

bob::learn::em::Gaussian& bob::learn::em::Gaussian::operator=(const bob::learn::em::Gaussian &other) {
  if(this != &other)
    copy(other);

  return *this;
}

bool bob::learn::em::Gaussian::operator==(const bob::learn::em::Gaussian& b) const
{
  return (bob::core::array::isEqual(m_mean, b.m_mean) &&
          bob::core::array::isEqual(m_variance, b.m_variance) &&
          bob::core::array::isEqual(m_variance_thresholds, b.m_variance_thresholds));
}

bool bob::learn::em::Gaussian::operator!=(const bob::learn::em::Gaussian& b) const {
  return !(this->operator==(b));
}

bool bob::learn::em::Gaussian::is_similar_to(const bob::learn::em::Gaussian& b,
  const double r_epsilon, const double a_epsilon) const
{
  return (bob::core::array::isClose(m_mean, b.m_mean, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_variance, b.m_variance, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_variance_thresholds, b.m_variance_thresholds, r_epsilon, a_epsilon));
}

void bob::learn::em::Gaussian::copy(const bob::learn::em::Gaussian& other) {
  m_n_inputs = other.m_n_inputs;

  m_mean.resize(m_n_inputs);
  m_mean = other.m_mean;

  m_variance.resize(m_n_inputs);
  m_variance = other.m_variance;

  m_variance_thresholds.resize(m_n_inputs);
  m_variance_thresholds = other.m_variance_thresholds;

  m_n_log2pi = other.m_n_log2pi;
  m_g_norm = other.m_g_norm;
}


void bob::learn::em::Gaussian::setNInputs(const size_t n_inputs) {
  resize(n_inputs);
}

void bob::learn::em::Gaussian::resize(const size_t n_inputs) {
  m_n_inputs = n_inputs;
  m_mean.resize(m_n_inputs);
  m_mean = 0;
  m_variance.resize(m_n_inputs);
  m_variance = 1;
  m_variance_thresholds.resize(m_n_inputs);
  m_variance_thresholds = 0;

  // Re-compute g_norm, because m_n_inputs and m_variance
  // have changed
  preComputeNLog2Pi();
  preComputeConstants();
}

void bob::learn::em::Gaussian::setMean(const blitz::Array<double,1> &mean) {
  // Check and set
  bob::core::array::assertSameShape(m_mean, mean);
  m_mean = mean;
}

void bob::learn::em::Gaussian::setVariance(const blitz::Array<double,1> &variance) {
  // Check and set
  bob::core::array::assertSameShape(m_variance, variance);
  m_variance = variance;

  // Variance flooring
  applyVarianceThresholds();
}

void bob::learn::em::Gaussian::setVarianceThresholds(const blitz::Array<double,1> &variance_thresholds) {
  // Check and set
  bob::core::array::assertSameShape(m_variance_thresholds, variance_thresholds);
  m_variance_thresholds = variance_thresholds;

  // Variance flooring
  applyVarianceThresholds();
}

void bob::learn::em::Gaussian::setVarianceThresholds(const double value) {
  blitz::Array<double,1> variance_thresholds(m_n_inputs);
  variance_thresholds = value;
  setVarianceThresholds(variance_thresholds);
}

void bob::learn::em::Gaussian::applyVarianceThresholds() {
   // Apply variance flooring threshold
  m_variance = blitz::where( m_variance < m_variance_thresholds, m_variance_thresholds, m_variance);

  // Re-compute g_norm, because m_variance has changed
  preComputeConstants();
}

double bob::learn::em::Gaussian::logLikelihood(const blitz::Array<double,1> &x) const {
  // Check
  bob::core::array::assertSameShape(x, m_mean);
  return logLikelihood_(x);
}

double bob::learn::em::Gaussian::logLikelihood_(const blitz::Array<double,1> &x) const {
  double z = blitz::sum(blitz::pow2(x - m_mean) / m_variance);
  // Log Likelihood
  return (-0.5 * (m_g_norm + z));
}

void bob::learn::em::Gaussian::preComputeNLog2Pi() {
  m_n_log2pi = m_n_inputs * bob::math::Log::Log2Pi;
}

void bob::learn::em::Gaussian::preComputeConstants() {
  m_g_norm = m_n_log2pi + blitz::sum(blitz::log(m_variance));
}

void bob::learn::em::Gaussian::save(bob::io::base::HDF5File& config) const {
  config.setArray("m_mean", m_mean);
  config.setArray("m_variance", m_variance);
  config.setArray("m_variance_thresholds", m_variance_thresholds);
  config.set("g_norm", m_g_norm);
  int64_t v = static_cast<int64_t>(m_n_inputs);
  config.set("m_n_inputs", v);
}

void bob::learn::em::Gaussian::load(bob::io::base::HDF5File& config) {
  int64_t v = config.read<int64_t>("m_n_inputs");
  m_n_inputs = static_cast<size_t>(v);

  m_mean.resize(m_n_inputs);
  m_variance.resize(m_n_inputs);
  m_variance_thresholds.resize(m_n_inputs);

  config.readArray("m_mean", m_mean);
  config.readArray("m_variance", m_variance);
  config.readArray("m_variance_thresholds", m_variance_thresholds);

  preComputeNLog2Pi();
  m_g_norm = config.read<double>("g_norm");
}

namespace bob { namespace learn { namespace em {
  std::ostream& operator<<(std::ostream& os, const Gaussian& g) {
    os << "Mean = " << g.m_mean << std::endl;
    os << "Variance = " << g.m_variance << std::endl;
    return os;
  }
} } }

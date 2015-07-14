/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.learn.em/GMMMachine.h>
#include <bob.core/assert.h>
#include <bob.math/log.h>

bob::learn::em::GMMMachine::GMMMachine(): m_gaussians(0) {
  resize(0,0);
}

bob::learn::em::GMMMachine::GMMMachine(const size_t n_gaussians, const size_t n_inputs):
  m_gaussians(0)
{
  resize(n_gaussians,n_inputs);
}

bob::learn::em::GMMMachine::GMMMachine(bob::io::base::HDF5File& config):
  m_gaussians(0)
{
  load(config);
}

bob::learn::em::GMMMachine::GMMMachine(const GMMMachine& other)  
{
  copy(other);
}


bob::learn::em::GMMMachine& bob::learn::em::GMMMachine::operator=(const bob::learn::em::GMMMachine &other) {
  // protect against invalid self-assignment
  if (this != &other)
    copy(other);

  // by convention, always return *this
  return *this;
}

bool bob::learn::em::GMMMachine::operator==(const bob::learn::em::GMMMachine& b) const
{
  if (m_n_gaussians != b.m_n_gaussians || m_n_inputs != b.m_n_inputs ||
      !bob::core::array::isEqual(m_weights, b.m_weights))
    return false;

  for(size_t i=0; i<m_n_gaussians; ++i) {
    if(!(*(m_gaussians[i]) == *(b.m_gaussians[i])))
      return false;
  }

  return true;
}

bool bob::learn::em::GMMMachine::operator!=(const bob::learn::em::GMMMachine& b) const {
  return !(this->operator==(b));
}

bool bob::learn::em::GMMMachine::is_similar_to(const bob::learn::em::GMMMachine& b,
  const double r_epsilon, const double a_epsilon) const
{
  if (m_n_gaussians != b.m_n_gaussians || m_n_inputs != b.m_n_inputs ||
      !bob::core::array::isClose(m_weights, b.m_weights, r_epsilon, a_epsilon))
    return false;

  for (size_t i = 0; i < m_n_gaussians; ++i)
    if (!m_gaussians[i]->is_similar_to(*b.m_gaussians[i], r_epsilon, a_epsilon))
      return false;

  return true;
}

void bob::learn::em::GMMMachine::copy(const GMMMachine& other) {
  m_n_gaussians = other.m_n_gaussians;
  m_n_inputs = other.m_n_inputs;

  // Initialise weights
  m_weights.resize(m_n_gaussians);
  m_weights = other.m_weights;

  // Initialise Gaussians
  m_gaussians.clear();
  for(size_t i=0; i<m_n_gaussians; ++i) {
    boost::shared_ptr<bob::learn::em::Gaussian> g(new bob::learn::em::Gaussian(*(other.m_gaussians[i])));
    m_gaussians.push_back(g);
  }

  // Initialise cache
  initCache();
}


bob::learn::em::GMMMachine::~GMMMachine() { 
}


/////////////////////
// Setters 
////////////////////

void bob::learn::em::GMMMachine::setWeights(const blitz::Array<double,1> &weights) {
  bob::core::array::assertSameShape(weights, m_weights);
  m_weights = weights;
  recomputeLogWeights();
}

void bob::learn::em::GMMMachine::recomputeLogWeights() const
{
  m_cache_log_weights = blitz::log(m_weights);
}

void bob::learn::em::GMMMachine::setMeans(const blitz::Array<double,2> &means) {
  bob::core::array::assertSameDimensionLength(means.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(means.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    m_gaussians[i]->updateMean() = means(i,blitz::Range::all());
  m_cache_supervector = false;
}

void bob::learn::em::GMMMachine::setMeanSupervector(const blitz::Array<double,1> &mean_supervector) {
  bob::core::array::assertSameDimensionLength(mean_supervector.extent(0), m_n_gaussians*m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    m_gaussians[i]->updateMean() = mean_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1));
  m_cache_supervector = false;
}


void bob::learn::em::GMMMachine::setVariances(const blitz::Array<double, 2 >& variances) {
  bob::core::array::assertSameDimensionLength(variances.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(variances.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i]->updateVariance() = variances(i,blitz::Range::all());
    m_gaussians[i]->applyVarianceThresholds();
  }
  m_cache_supervector = false;
}

void bob::learn::em::GMMMachine::setVarianceSupervector(const blitz::Array<double,1> &variance_supervector) {
  bob::core::array::assertSameDimensionLength(variance_supervector.extent(0), m_n_gaussians*m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i]->updateVariance() = variance_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1));
    m_gaussians[i]->applyVarianceThresholds();
  }
  m_cache_supervector = false;
}

void bob::learn::em::GMMMachine::setVarianceThresholds(const double value) {
  for(size_t i=0; i<m_n_gaussians; ++i)
    m_gaussians[i]->setVarianceThresholds(value);
  m_cache_supervector = false;
}

void bob::learn::em::GMMMachine::setVarianceThresholds(blitz::Array<double, 1> variance_thresholds) {
  bob::core::array::assertSameDimensionLength(variance_thresholds.extent(0), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    m_gaussians[i]->setVarianceThresholds(variance_thresholds);
  m_cache_supervector = false;
}

void bob::learn::em::GMMMachine::setVarianceThresholds(const blitz::Array<double, 2>& variance_thresholds) {
  bob::core::array::assertSameDimensionLength(variance_thresholds.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(variance_thresholds.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    m_gaussians[i]->setVarianceThresholds(variance_thresholds(i,blitz::Range::all()));
  m_cache_supervector = false;
}

/////////////////////
// Getters 
////////////////////

const blitz::Array<double,2> bob::learn::em::GMMMachine::getMeans() const {

  blitz::Array<double,2> means(m_n_gaussians,m_n_inputs);  
  for(size_t i=0; i<m_n_gaussians; ++i)
    means(i,blitz::Range::all()) = m_gaussians[i]->getMean();
    
  return means;
}

const blitz::Array<double,2> bob::learn::em::GMMMachine::getVariances() const{
  
  blitz::Array<double,2> variances(m_n_gaussians,m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    variances(i,blitz::Range::all()) = m_gaussians[i]->getVariance();

  return variances;
}


const blitz::Array<double,2>  bob::learn::em::GMMMachine::getVarianceThresholds() const {
  //bob::core::array::assertSameDimensionLength(variance_thresholds.extent(0), m_n_gaussians);
  //bob::core::array::assertSameDimensionLength(variance_thresholds.extent(1), m_n_inputs);
  blitz::Array<double, 2> variance_thresholds(m_n_gaussians, m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    variance_thresholds(i,blitz::Range::all()) = m_gaussians[i]->getVarianceThresholds();

  return variance_thresholds;
}


/////////////////////
// Methods
////////////////////


void bob::learn::em::GMMMachine::resize(const size_t n_gaussians, const size_t n_inputs) {
  m_n_gaussians = n_gaussians;
  m_n_inputs = n_inputs;

  // Initialise weights
  m_weights.resize(m_n_gaussians);
  m_weights = 1.0 / m_n_gaussians;

  // Initialise Gaussians
  m_gaussians.clear();
  for(size_t i=0; i<m_n_gaussians; ++i)
    m_gaussians.push_back(boost::shared_ptr<bob::learn::em::Gaussian>(new bob::learn::em::Gaussian(n_inputs)));

  // Initialise cache arrays
  initCache();
}

double bob::learn::em::GMMMachine::logLikelihood(const blitz::Array<double, 1> &x,
  blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const
{
  // Check dimension
  bob::core::array::assertSameDimensionLength(log_weighted_gaussian_likelihoods.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_n_inputs);
  return logLikelihood_(x,log_weighted_gaussian_likelihoods);
}


double bob::learn::em::GMMMachine::logLikelihood_(const blitz::Array<double, 1> &x,
  blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const
{
  // Initialise variables
  double log_likelihood = bob::math::Log::LogZero;

  // Accumulate the weighted log likelihoods from each Gaussian
  for(size_t i=0; i<m_n_gaussians; ++i) {
    double l = m_cache_log_weights(i) + m_gaussians[i]->logLikelihood_(x);
    log_weighted_gaussian_likelihoods(i) = l;
    log_likelihood = bob::math::Log::logAdd(log_likelihood, l);
  }

  // Return log(p(x|GMMMachine))
  return log_likelihood;
}


double bob::learn::em::GMMMachine::logLikelihood(const blitz::Array<double, 2> &x) const {
  // Check dimension
  bob::core::array::assertSameDimensionLength(x.extent(1), m_n_inputs);
  // Call the other logLikelihood_ (overloaded) function


  double sum_ll = 0;
  for (int i=0; i<x.extent(0); i++)
    sum_ll+= logLikelihood_(x(i,blitz::Range::all()));

  return sum_ll/x.extent(0);  
}



double bob::learn::em::GMMMachine::logLikelihood(const blitz::Array<double, 1> &x) const {
  // Check dimension
  bob::core::array::assertSameDimensionLength(x.extent(0), m_n_inputs);
  // Call the other logLikelihood_ (overloaded) function
  // (log_weighted_gaussian_likelihoods will be discarded)
  return logLikelihood_(x,m_cache_log_weighted_gaussian_likelihoods);
}



double bob::learn::em::GMMMachine::logLikelihood_(const blitz::Array<double, 1> &x) const {
  // Call the other logLikelihood (overloaded) function
  // (log_weighted_gaussian_likelihoods will be discarded)
  return logLikelihood_(x,m_cache_log_weighted_gaussian_likelihoods);
}

void bob::learn::em::GMMMachine::accStatistics(const blitz::Array<double,2>& input,
    bob::learn::em::GMMStats& stats) const {
  // iterate over data
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<input.extent(0); ++i) {
    // Get example
    blitz::Array<double,1> x(input(i,a));
    // Accumulate statistics
    accStatistics(x,stats);
  }
}

void bob::learn::em::GMMMachine::accStatistics_(const blitz::Array<double,2>& input, bob::learn::em::GMMStats& stats) const {
  // iterate over data
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<input.extent(0); ++i) {
    // Get example
    blitz::Array<double,1> x(input(i, a));
    // Accumulate statistics
    accStatistics_(x,stats);
  }
}

void bob::learn::em::GMMMachine::accStatistics(const blitz::Array<double, 1>& x, bob::learn::em::GMMStats& stats) const {
  // check GMMStats size
  bob::core::array::assertSameDimensionLength(stats.sumPx.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(stats.sumPx.extent(1), m_n_inputs);

  // Calculate Gaussian and GMM likelihoods
  // - m_cache_log_weighted_gaussian_likelihoods(i) = log(weight_i*p(x|gaussian_i))
  // - log_likelihood = log(sum_i(weight_i*p(x|gaussian_i)))
  double log_likelihood = logLikelihood(x, m_cache_log_weighted_gaussian_likelihoods);

  accStatisticsInternal(x, stats, log_likelihood);
}

void bob::learn::em::GMMMachine::accStatistics_(const blitz::Array<double, 1>& x, bob::learn::em::GMMStats& stats) const {
  // Calculate Gaussian and GMM likelihoods
  // - m_cache_log_weighted_gaussian_likelihoods(i) = log(weight_i*p(x|gaussian_i))
  // - log_likelihood = log(sum_i(weight_i*p(x|gaussian_i)))
  double log_likelihood = logLikelihood_(x, m_cache_log_weighted_gaussian_likelihoods);

  accStatisticsInternal(x, stats, log_likelihood);
}

void bob::learn::em::GMMMachine::accStatisticsInternal(const blitz::Array<double, 1>& x,
  bob::learn::em::GMMStats& stats, const double log_likelihood) const
{
  // Calculate responsibilities
  m_cache_P = blitz::exp(m_cache_log_weighted_gaussian_likelihoods - log_likelihood);

  // Accumulate statistics
  // - total likelihood
  stats.log_likelihood += log_likelihood;

  // - number of samples
  stats.T++;

  // - responsibilities
  stats.n += m_cache_P;

  // - first order stats
  blitz::firstIndex i;
  blitz::secondIndex j;

  m_cache_Px = m_cache_P(i) * x(j);

  stats.sumPx += m_cache_Px;

  // - second order stats
  stats.sumPxx += (m_cache_Px(i,j) * x(j));
}

boost::shared_ptr<bob::learn::em::Gaussian> bob::learn::em::GMMMachine::getGaussian(const size_t i) {
  if (i>=m_n_gaussians) {
    throw std::runtime_error("getGaussian(): index out of bounds");
  }
  return m_gaussians[i];
}

void bob::learn::em::GMMMachine::save(bob::io::base::HDF5File& config) const {
  int64_t v = static_cast<int64_t>(m_n_gaussians);
  config.set("m_n_gaussians", v);
  v = static_cast<int64_t>(m_n_inputs);
  config.set("m_n_inputs", v);

  for(size_t i=0; i<m_n_gaussians; ++i) {
    std::ostringstream oss;
    oss << "m_gaussians" << i;

    if (!config.hasGroup(oss.str())) config.createGroup(oss.str());
    config.cd(oss.str());
    m_gaussians[i]->save(config);
    config.cd("..");
  }

  config.setArray("m_weights", m_weights);
}

void bob::learn::em::GMMMachine::load(bob::io::base::HDF5File& config) {
  int64_t v;
  v = config.read<int64_t>("m_n_gaussians");
  m_n_gaussians = static_cast<size_t>(v);
  v = config.read<int64_t>("m_n_inputs");
  m_n_inputs = static_cast<size_t>(v);

  m_gaussians.clear();
  for(size_t i=0; i<m_n_gaussians; ++i) {
    m_gaussians.push_back(boost::shared_ptr<bob::learn::em::Gaussian>(new bob::learn::em::Gaussian(m_n_inputs)));
    std::ostringstream oss;
    oss << "m_gaussians" << i;
    config.cd(oss.str());
    m_gaussians[i]->load(config);
    config.cd("..");
  }

  m_weights.resize(m_n_gaussians);
  config.readArray("m_weights", m_weights);

  // Initialise cache
  initCache();
}

void bob::learn::em::GMMMachine::updateCacheSupervectors() const
{
  m_cache_mean_supervector.resize(m_n_gaussians*m_n_inputs);
  m_cache_variance_supervector.resize(m_n_gaussians*m_n_inputs);

  for(size_t i=0; i<m_n_gaussians; ++i) {
    blitz::Range range(i*m_n_inputs, (i+1)*m_n_inputs-1);
    m_cache_mean_supervector(range) = m_gaussians[i]->getMean();
    m_cache_variance_supervector(range) = m_gaussians[i]->getVariance();
  }
  m_cache_supervector = true;
}

void bob::learn::em::GMMMachine::initCache() const {
  // Initialise cache arrays
  m_cache_log_weights.resize(m_n_gaussians);
  recomputeLogWeights();
  m_cache_log_weighted_gaussian_likelihoods.resize(m_n_gaussians);
  m_cache_P.resize(m_n_gaussians);
  m_cache_Px.resize(m_n_gaussians,m_n_inputs);
  m_cache_supervector = false;
}

void bob::learn::em::GMMMachine::reloadCacheSupervectors() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
}

const blitz::Array<double,1>& bob::learn::em::GMMMachine::getMeanSupervector() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
  return m_cache_mean_supervector;
}

const blitz::Array<double,1>& bob::learn::em::GMMMachine::getVarianceSupervector() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
  return m_cache_variance_supervector;
}

namespace bob { namespace learn { namespace em {
  std::ostream& operator<<(std::ostream& os, const GMMMachine& machine) {
    os << "Weights = " << machine.m_weights << std::endl;
    for(size_t i=0; i < machine.m_n_gaussians; ++i) {
      os << "Gaussian " << i << ": " << std::endl << *(machine.m_gaussians[i]);
    }

    return os;
  }
} } }

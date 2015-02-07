/**
 * @date Tue Jan 27 16:47:00 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */


#include <bob.learn.em/JFAMachine.h>
#include <bob.core/array_copy.h>
#include <bob.math/linear.h>
#include <bob.math/inv.h>
#include <bob.learn.em/LinearScoring.h>
#include <limits>


//////////////////// JFAMachine ////////////////////
bob::learn::em::JFAMachine::JFAMachine():
  m_y(1), m_z(1)
{
  resizeTmp();
}

bob::learn::em::JFAMachine::JFAMachine(const boost::shared_ptr<bob::learn::em::JFABase> jfa_base):
  m_jfa_base(jfa_base),
  m_y(jfa_base->getDimRv()), m_z(jfa_base->getSupervectorLength())
{
  if (!m_jfa_base->getUbm()) throw std::runtime_error("No UBM was set in the JFA machine.");
  updateCache();
  resizeTmp();
}


bob::learn::em::JFAMachine::JFAMachine(const bob::learn::em::JFAMachine& other):
  m_jfa_base(other.m_jfa_base),
  m_y(bob::core::array::ccopy(other.m_y)),
  m_z(bob::core::array::ccopy(other.m_z))
{
  updateCache();
  resizeTmp();
}

bob::learn::em::JFAMachine::JFAMachine(bob::io::base::HDF5File& config)
{
  load(config);
}

bob::learn::em::JFAMachine::~JFAMachine() {
}

bob::learn::em::JFAMachine&
bob::learn::em::JFAMachine::operator=(const bob::learn::em::JFAMachine& other)
{
  if (this != &other)
  {
    m_jfa_base = other.m_jfa_base;
    m_y.reference(bob::core::array::ccopy(other.m_y));
    m_z.reference(bob::core::array::ccopy(other.m_z));
  }
  return *this;
}

bool bob::learn::em::JFAMachine::operator==(const bob::learn::em::JFAMachine& other) const
{
  return (*m_jfa_base == *(other.m_jfa_base) &&
          bob::core::array::isEqual(m_y, other.m_y) &&
          bob::core::array::isEqual(m_z, other.m_z));
}

bool bob::learn::em::JFAMachine::operator!=(const bob::learn::em::JFAMachine& b) const
{
  return !(this->operator==(b));
}


bool bob::learn::em::JFAMachine::is_similar_to(const bob::learn::em::JFAMachine& b,
    const double r_epsilon, const double a_epsilon) const
{
  return (m_jfa_base->is_similar_to(*(b.m_jfa_base), r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_y, b.m_y, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_z, b.m_z, r_epsilon, a_epsilon));
}

void bob::learn::em::JFAMachine::save(bob::io::base::HDF5File& config) const
{
  config.setArray("y", m_y);
  config.setArray("z", m_z);
}

void bob::learn::em::JFAMachine::load(bob::io::base::HDF5File& config)
{
  //reads all data directly into the member variables
  blitz::Array<double,1> y = config.readArray<double,1>("y");
  blitz::Array<double,1> z = config.readArray<double,1>("z");
  if (!m_jfa_base)
  {
    m_y.resize(y.extent(0));
    m_z.resize(z.extent(0));
  }
  setY(y);
  setZ(z);
  // update cache
  updateCache();
  resizeTmp();
}


void bob::learn::em::JFAMachine::setY(const blitz::Array<double,1>& y)
{
  if(y.extent(0) != m_y.extent(0)) { //checks dimension
    boost::format m("size of input vector `y' (%d) does not match the expected size (%d)");
    m % y.extent(0) % m_y.extent(0);
    throw std::runtime_error(m.str());
  }
  m_y.reference(bob::core::array::ccopy(y));
  // update cache
  updateCache();
}

void bob::learn::em::JFAMachine::setZ(const blitz::Array<double,1>& z)
{
  if(z.extent(0) != m_z.extent(0)) { //checks dimension
    boost::format m("size of input vector `z' (%d) does not match the expected size (%d)");
    m % z.extent(0) % m_z.extent(0);
    throw std::runtime_error(m.str());
  }
  m_z.reference(bob::core::array::ccopy(z));
  // update cache
  updateCache();
}

void bob::learn::em::JFAMachine::setJFABase(const boost::shared_ptr<bob::learn::em::JFABase> jfa_base)
{
  if (!jfa_base->getUbm())
    throw std::runtime_error("No UBM was set in the JFA machine.");
  m_jfa_base = jfa_base;
  // Resize variables
  resize();
}

void bob::learn::em::JFAMachine::resize()
{
  m_y.resizeAndPreserve(getDimRv());
  m_z.resizeAndPreserve(getSupervectorLength());
  updateCache();
  resizeTmp();
}

void bob::learn::em::JFAMachine::resizeTmp()
{
  if (m_jfa_base)
  {
    m_tmp_Ux.resize(getSupervectorLength());
  }
}

void bob::learn::em::JFAMachine::updateCache()
{
  if (m_jfa_base)
  {
    // m + Vy + Dz
    m_cache_mVyDz.resize(getSupervectorLength());
    bob::math::prod(m_jfa_base->getV(), m_y, m_cache_mVyDz);
    m_cache_mVyDz += m_jfa_base->getD()*m_z + m_jfa_base->getUbm()->getMeanSupervector();
    m_cache_x.resize(getDimRu());
  }
}

void bob::learn::em::JFAMachine::estimateUx(const bob::learn::em::GMMStats& gmm_stats,
  blitz::Array<double,1>& Ux)
{
  estimateX(gmm_stats, m_cache_x);
  bob::math::prod(m_jfa_base->getU(), m_cache_x, Ux);
}

double bob::learn::em::JFAMachine::forward(const bob::learn::em::GMMStats& input)
{
  return forward_(input);
}

double bob::learn::em::JFAMachine::forward(const bob::learn::em::GMMStats& gmm_stats,
  const blitz::Array<double,1>& Ux)
{
  // Checks that a Base machine has been set
  if (!m_jfa_base) throw std::runtime_error("No UBM was set in the JFA machine.");

  return bob::learn::em::linearScoring(m_cache_mVyDz,
            m_jfa_base->getUbm()->getMeanSupervector(), m_jfa_base->getUbm()->getVarianceSupervector(),
            gmm_stats, Ux, true);
}

double bob::learn::em::JFAMachine::forward_(const bob::learn::em::GMMStats& input)
{
  // Checks that a Base machine has been set
  if (!m_jfa_base) throw std::runtime_error("No UBM was set in the JFA machine.");

  // Ux and GMMStats
  estimateX(input, m_cache_x);
  bob::math::prod(m_jfa_base->getU(), m_cache_x, m_tmp_Ux);

  return bob::learn::em::linearScoring(m_cache_mVyDz,
            m_jfa_base->getUbm()->getMeanSupervector(), m_jfa_base->getUbm()->getVarianceSupervector(),
            input, m_tmp_Ux, true);
}


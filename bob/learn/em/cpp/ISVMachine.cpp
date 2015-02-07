/**
 * @date Tue Jan 27 16:06:00 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */


#include <bob.learn.em/ISVMachine.h>
#include <bob.core/array_copy.h>
#include <bob.math/linear.h>
#include <bob.math/inv.h>
#include <bob.learn.em/LinearScoring.h>
#include <limits>


//////////////////// ISVMachine ////////////////////
bob::learn::em::ISVMachine::ISVMachine():
  m_z(1)
{
  resizeTmp();
}

bob::learn::em::ISVMachine::ISVMachine(const boost::shared_ptr<bob::learn::em::ISVBase> isv_base):
  m_isv_base(isv_base),
  m_z(isv_base->getSupervectorLength())
{
  if (!m_isv_base->getUbm())
    throw std::runtime_error("No UBM was set in the JFA machine.");
  updateCache();
  resizeTmp();
}


bob::learn::em::ISVMachine::ISVMachine(const bob::learn::em::ISVMachine& other):
  m_isv_base(other.m_isv_base),
  m_z(bob::core::array::ccopy(other.m_z))
{
  updateCache();
  resizeTmp();
}

bob::learn::em::ISVMachine::ISVMachine(bob::io::base::HDF5File& config)
{
  load(config);
}

bob::learn::em::ISVMachine::~ISVMachine() {
}

bob::learn::em::ISVMachine&
bob::learn::em::ISVMachine::operator=(const bob::learn::em::ISVMachine& other)
{
  if (this != &other)
  {
    m_isv_base = other.m_isv_base;
    m_z.reference(bob::core::array::ccopy(other.m_z));
  }
  return *this;
}

bool bob::learn::em::ISVMachine::operator==(const bob::learn::em::ISVMachine& other) const
{
  return (*m_isv_base == *(other.m_isv_base) &&
          bob::core::array::isEqual(m_z, other.m_z));
}

bool bob::learn::em::ISVMachine::operator!=(const bob::learn::em::ISVMachine& b) const
{
  return !(this->operator==(b));
}


bool bob::learn::em::ISVMachine::is_similar_to(const bob::learn::em::ISVMachine& b,
    const double r_epsilon, const double a_epsilon) const
{
  return (m_isv_base->is_similar_to(*(b.m_isv_base), r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_z, b.m_z, r_epsilon, a_epsilon));
}

void bob::learn::em::ISVMachine::save(bob::io::base::HDF5File& config) const
{
  config.setArray("z", m_z);
}

void bob::learn::em::ISVMachine::load(bob::io::base::HDF5File& config)
{
  //reads all data directly into the member variables
  blitz::Array<double,1> z = config.readArray<double,1>("z");
  if (!m_isv_base)
    m_z.resize(z.extent(0));
  setZ(z);
  // update cache
  updateCache();
  resizeTmp();
}

void bob::learn::em::ISVMachine::setZ(const blitz::Array<double,1>& z)
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

void bob::learn::em::ISVMachine::setISVBase(const boost::shared_ptr<bob::learn::em::ISVBase> isv_base)
{
  if (!isv_base->getUbm())
    throw std::runtime_error("No UBM was set in the JFA machine.");
  m_isv_base = isv_base;
  // Resize variables
  resize();
}

void bob::learn::em::ISVMachine::resize()
{
  m_z.resizeAndPreserve(getSupervectorLength());
  updateCache();
  resizeTmp();
}

void bob::learn::em::ISVMachine::resizeTmp()
{
  if (m_isv_base)
  {
    m_tmp_Ux.resize(getSupervectorLength());
  }
}

void bob::learn::em::ISVMachine::updateCache()
{
  if (m_isv_base)
  {
    // m + Dz
    m_cache_mDz.resize(getSupervectorLength());
    m_cache_mDz = m_isv_base->getD()*m_z + m_isv_base->getUbm()->getMeanSupervector();
    m_cache_x.resize(getDimRu());
  }
}

void bob::learn::em::ISVMachine::estimateUx(const bob::learn::em::GMMStats& gmm_stats,
  blitz::Array<double,1>& Ux)
{
  estimateX(gmm_stats, m_cache_x);
  bob::math::prod(m_isv_base->getU(), m_cache_x, Ux);
}

double bob::learn::em::ISVMachine::forward(const bob::learn::em::GMMStats& input)
{
  return forward_(input);
}

double bob::learn::em::ISVMachine::forward(const bob::learn::em::GMMStats& gmm_stats,
  const blitz::Array<double,1>& Ux)
{
  // Checks that a Base machine has been set
  if (!m_isv_base) throw std::runtime_error("No UBM was set in the JFA machine.");

  return bob::learn::em::linearScoring(m_cache_mDz,
            m_isv_base->getUbm()->getMeanSupervector(), m_isv_base->getUbm()->getVarianceSupervector(),
            gmm_stats, Ux, true);
}

double bob::learn::em::ISVMachine::forward_(const bob::learn::em::GMMStats& input)
{
  // Checks that a Base machine has been set
  if(!m_isv_base) throw std::runtime_error("No UBM was set in the JFA machine.");

  // Ux and GMMStats
  estimateX(input, m_cache_x);
  bob::math::prod(m_isv_base->getU(), m_cache_x, m_tmp_Ux);

  return bob::learn::em::linearScoring(m_cache_mDz,
            m_isv_base->getUbm()->getMeanSupervector(), m_isv_base->getUbm()->getVarianceSupervector(),
            input, m_tmp_Ux, true);
}


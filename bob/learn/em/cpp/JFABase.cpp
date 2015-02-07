/**
 * @date Tue Jan 27 15:54:00 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */


#include <bob.learn.em/JFABase.h>
#include <bob.core/array_copy.h>
#include <bob.math/linear.h>
#include <bob.math/inv.h>
#include <bob.learn.em/LinearScoring.h>
#include <limits>


//////////////////// JFABase ////////////////////
bob::learn::em::JFABase::JFABase()
{
}

bob::learn::em::JFABase::JFABase(const boost::shared_ptr<bob::learn::em::GMMMachine> ubm,
    const size_t ru, const size_t rv):
  m_base(ubm, ru, rv)
{
}

bob::learn::em::JFABase::JFABase(const bob::learn::em::JFABase& other):
  m_base(other.m_base)
{
}


bob::learn::em::JFABase::JFABase(bob::io::base::HDF5File& config)
{
  load(config);
}

bob::learn::em::JFABase::~JFABase() {
}

void bob::learn::em::JFABase::save(bob::io::base::HDF5File& config) const
{
  config.setArray("U", m_base.getU());
  config.setArray("V", m_base.getV());
  config.setArray("d", m_base.getD());
}

void bob::learn::em::JFABase::load(bob::io::base::HDF5File& config)
{
  //reads all data directly into the member variables
  blitz::Array<double,2> U = config.readArray<double,2>("U");
  blitz::Array<double,2> V = config.readArray<double,2>("V");
  blitz::Array<double,1> d = config.readArray<double,1>("d");
  const int ru = U.extent(1);
  const int rv = V.extent(1);
  if (!m_base.getUbm())
    m_base.resize(ru, rv, U.extent(0));
  else
    m_base.resize(ru, rv);
  m_base.setU(U);
  m_base.setV(V);
  m_base.setD(d);
}

bob::learn::em::JFABase&
bob::learn::em::JFABase::operator=(const bob::learn::em::JFABase& other)
{
  if (this != &other)
  {
    m_base = other.m_base;
  }
  return *this;
}

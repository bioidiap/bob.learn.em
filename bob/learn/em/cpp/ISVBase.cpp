/**
 * @date Tue Jan 27 16:02:00 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */


#include <bob.learn.em/ISVBase.h>
#include <bob.core/array_copy.h>
#include <bob.math/linear.h>
#include <bob.math/inv.h>
#include <bob.learn.em/LinearScoring.h>
#include <limits>


//////////////////// ISVBase ////////////////////
bob::learn::em::ISVBase::ISVBase()
{
}

bob::learn::em::ISVBase::ISVBase(const boost::shared_ptr<bob::learn::em::GMMMachine> ubm,
    const size_t ru):
  m_base(ubm, ru, 1)
{
  blitz::Array<double,2>& V = m_base.updateV();
  V = 0;
}

bob::learn::em::ISVBase::ISVBase(const bob::learn::em::ISVBase& other):
  m_base(other.m_base)
{
}


bob::learn::em::ISVBase::ISVBase(bob::io::base::HDF5File& config)
{
  load(config);
}

bob::learn::em::ISVBase::~ISVBase() {
}

void bob::learn::em::ISVBase::save(bob::io::base::HDF5File& config) const
{
  config.setArray("U", m_base.getU());
  config.setArray("d", m_base.getD());
}

void bob::learn::em::ISVBase::load(bob::io::base::HDF5File& config)
{
  //reads all data directly into the member variables
  blitz::Array<double,2> U = config.readArray<double,2>("U");
  blitz::Array<double,1> d = config.readArray<double,1>("d");
  const int ru = U.extent(1);
  if (!m_base.getUbm())
    m_base.resize(ru, 1, U.extent(0));
  else
    m_base.resize(ru, 1);
  m_base.setU(U);
  m_base.setD(d);
  blitz::Array<double,2>& V = m_base.updateV();
  V = 0;
}

bob::learn::em::ISVBase&
bob::learn::em::ISVBase::operator=(const bob::learn::em::ISVBase& other)
{
  if (this != &other)
  {
    m_base = other.m_base;
  }
  return *this;
}


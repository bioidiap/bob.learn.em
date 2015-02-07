/**
 * @date Tue Jan 27 15:51:15 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */


#include <bob.learn.em/FABase.h>
#include <bob.core/array_copy.h>
#include <bob.math/linear.h>
#include <bob.math/inv.h>
#include <limits>


//////////////////// FABase ////////////////////
bob::learn::em::FABase::FABase():
  m_ubm(boost::shared_ptr<bob::learn::em::GMMMachine>()), m_ru(1), m_rv(1),
  m_U(0,1), m_V(0,1), m_d(0)
{}

bob::learn::em::FABase::FABase(const boost::shared_ptr<bob::learn::em::GMMMachine> ubm,
    const size_t ru, const size_t rv):
  m_ubm(ubm), m_ru(ru), m_rv(rv),
  m_U(getSupervectorLength(),ru), m_V(getSupervectorLength(),rv), m_d(getSupervectorLength())
{
  if (ru < 1) {
    boost::format m("value for parameter `ru' (%lu) cannot be smaller than 1");
    m % ru;
    throw std::runtime_error(m.str());
  }
  if (rv < 1) {
    boost::format m("value for parameter `rv' (%lu) cannot be smaller than 1");
    m % ru;
    throw std::runtime_error(m.str());
  }
  updateCache();
}

bob::learn::em::FABase::FABase(const bob::learn::em::FABase& other):
  m_ubm(other.m_ubm), m_ru(other.m_ru), m_rv(other.m_rv),
  m_U(bob::core::array::ccopy(other.m_U)),
  m_V(bob::core::array::ccopy(other.m_V)),
  m_d(bob::core::array::ccopy(other.m_d))
{
  updateCache();
}

bob::learn::em::FABase::~FABase() {
}

bob::learn::em::FABase& bob::learn::em::FABase::operator=
(const bob::learn::em::FABase& other)
{
  if (this != &other)
  {
    m_ubm = other.m_ubm;
    m_ru = other.m_ru;
    m_rv = other.m_rv;
    m_U.reference(bob::core::array::ccopy(other.m_U));
    m_V.reference(bob::core::array::ccopy(other.m_V));
    m_d.reference(bob::core::array::ccopy(other.m_d));

    updateCache();
  }
  return *this;
}

bool bob::learn::em::FABase::operator==(const bob::learn::em::FABase& b) const
{
  return ( (((m_ubm && b.m_ubm) && *m_ubm == *(b.m_ubm)) || (!m_ubm && !b.m_ubm)) &&
          m_ru == b.m_ru && m_rv == b.m_rv &&
          bob::core::array::isEqual(m_U, b.m_U) &&
          bob::core::array::isEqual(m_V, b.m_V) &&
          bob::core::array::isEqual(m_d, b.m_d));
}

bool bob::learn::em::FABase::operator!=(const bob::learn::em::FABase& b) const
{
  return !(this->operator==(b));
}

bool bob::learn::em::FABase::is_similar_to(const bob::learn::em::FABase& b,
    const double r_epsilon, const double a_epsilon) const
{
  // TODO: update is_similar_to of the GMMMachine with the 2 epsilon's
  return (( ((m_ubm && b.m_ubm) && m_ubm->is_similar_to(*(b.m_ubm), a_epsilon)) ||
            (!m_ubm && !b.m_ubm) ) &&
          m_ru == b.m_ru && m_rv == b.m_rv &&
          bob::core::array::isClose(m_U, b.m_U, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_V, b.m_V, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_d, b.m_d, r_epsilon, a_epsilon));
}

void bob::learn::em::FABase::resize(const size_t ru, const size_t rv)
{
  if (ru < 1) {
    boost::format m("value for parameter `ru' (%lu) cannot be smaller than 1");
    m % ru;
    throw std::runtime_error(m.str());
  }
  if (rv < 1) {
    boost::format m("value for parameter `rv' (%lu) cannot be smaller than 1");
    m % ru;
    throw std::runtime_error(m.str());
  }

  m_ru = ru;
  m_rv = rv;
  m_U.resizeAndPreserve(m_U.extent(0), ru);
  m_V.resizeAndPreserve(m_V.extent(0), rv);

  updateCacheUbmUVD();
}

void bob::learn::em::FABase::resize(const size_t ru, const size_t rv, const size_t cd)
{
  if (ru < 1) {
    boost::format m("value for parameter `ru' (%lu) cannot be smaller than 1");
    m % ru;
    throw std::runtime_error(m.str());
  }
  if (rv < 1) {
    boost::format m("value for parameter `rv' (%lu) cannot be smaller than 1");
    m % ru;
    throw std::runtime_error(m.str());
  }

  if (!m_ubm || (m_ubm && getSupervectorLength() == cd))
  {
    m_ru = ru;
    m_rv = rv;
    m_U.resizeAndPreserve(cd, ru);
    m_V.resizeAndPreserve(cd, rv);
    m_d.resizeAndPreserve(cd);

    updateCacheUbmUVD();
  }
  else {
    boost::format m("value for parameter `cd' (%lu) is not set to %lu");
    m % cd % getSupervectorLength();
    throw std::runtime_error(m.str());
  }
}

void bob::learn::em::FABase::setUbm(const boost::shared_ptr<bob::learn::em::GMMMachine> ubm)
{
  m_ubm = ubm;
  m_U.resizeAndPreserve(getSupervectorLength(), m_ru);
  m_V.resizeAndPreserve(getSupervectorLength(), m_rv);
  m_d.resizeAndPreserve(getSupervectorLength());

  updateCache();
}

void bob::learn::em::FABase::setU(const blitz::Array<double,2>& U)
{
  if(U.extent(0) != m_U.extent(0)) { //checks dimension
    boost::format m("number of rows in parameter `U' (%d) does not match the expected size (%d)");
    m % U.extent(0) % m_U.extent(0);
    throw std::runtime_error(m.str());
  }
  if(U.extent(1) != m_U.extent(1)) { //checks dimension
    boost::format m("number of columns in parameter `U' (%d) does not match the expected size (%d)");
    m % U.extent(1) % m_U.extent(1);
    throw std::runtime_error(m.str());
  }
  m_U.reference(bob::core::array::ccopy(U));

  // update cache
  updateCacheUbmUVD();
}

void bob::learn::em::FABase::setV(const blitz::Array<double,2>& V)
{
  if(V.extent(0) != m_V.extent(0)) { //checks dimension
    boost::format m("number of rows in parameter `V' (%d) does not match the expected size (%d)");
    m % V.extent(0) % m_V.extent(0);
    throw std::runtime_error(m.str());
  }
  if(V.extent(1) != m_V.extent(1)) { //checks dimension
    boost::format m("number of columns in parameter `V' (%d) does not match the expected size (%d)");
    m % V.extent(1) % m_V.extent(1);
    throw std::runtime_error(m.str());
  }
  m_V.reference(bob::core::array::ccopy(V));
}

void bob::learn::em::FABase::setD(const blitz::Array<double,1>& d)
{
  if(d.extent(0) != m_d.extent(0)) { //checks dimension
    boost::format m("size of input vector `d' (%d) does not match the expected size (%d)");
    m % d.extent(0) % m_d.extent(0);
    throw std::runtime_error(m.str());
  }
  m_d.reference(bob::core::array::ccopy(d));
}


void bob::learn::em::FABase::updateCache()
{
  updateCacheUbm();
  updateCacheUbmUVD();
  resizeTmp();
}

void bob::learn::em::FABase::resizeTmp()
{
  m_tmp_IdPlusUSProdInv.resize(getDimRu(),getDimRu());
  m_tmp_Fn_x.resize(getSupervectorLength());
  m_tmp_ru.resize(getDimRu());
  m_tmp_ruD.resize(getDimRu(), getNInputs());
  m_tmp_ruru.resize(getDimRu(), getDimRu());
}

void bob::learn::em::FABase::updateCacheUbm()
{
  // Put supervectors in cache
  if (m_ubm)
  {
    m_cache_mean.resize(getSupervectorLength());
    m_cache_sigma.resize(getSupervectorLength());
    m_cache_mean  = m_ubm->getMeanSupervector();
    m_cache_sigma = m_ubm->getVarianceSupervector();
  }
}

void bob::learn::em::FABase::updateCacheUbmUVD()
{
  // Compute and put  U^{T}.diag(sigma)^{-1} in cache
  if (m_ubm)
  {
    blitz::firstIndex i;
    blitz::secondIndex j;
    m_cache_UtSigmaInv.resize(getDimRu(), getSupervectorLength());
    m_cache_UtSigmaInv = m_U(j,i) / m_cache_sigma(j); // Ut * diag(sigma)^-1
  }
}

void bob::learn::em::FABase::computeIdPlusUSProdInv(const bob::learn::em::GMMStats& gmm_stats,
  blitz::Array<double,2>& output) const
{
  // Computes (Id + U^T.Sigma^-1.U.N_{i,h}.U)^-1 =
  // (Id + sum_{c=1..C} N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c})^-1

  // Blitz compatibility: ugly fix (const_cast, as old blitz version does not
  // provide a non-const version of transpose())
  blitz::Array<double,2> Ut = const_cast<blitz::Array<double,2>&>(m_U).transpose(1,0);

  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Range rall = blitz::Range::all();

  bob::math::eye(m_tmp_ruru); // m_tmp_ruru = Id
  // Loop and add N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c} to m_tmp_ruru at each iteration
  const size_t dim_c = getNGaussians();
  const size_t dim_d = getNInputs();
  for(size_t c=0; c<dim_c; ++c) {
    blitz::Range rc(c*dim_d,(c+1)*dim_d-1);
    blitz::Array<double,2> Ut_c = Ut(rall,rc);
    blitz::Array<double,1> sigma_c = m_cache_sigma(rc);
    m_tmp_ruD = Ut_c(i,j) / sigma_c(j); // U_{c}^T.Sigma_{c}^-1
    blitz::Array<double,2> U_c = m_U(rc,rall);
    // Use m_cache_IdPlusUSProdInv as an intermediate array
    bob::math::prod(m_tmp_ruD, U_c, output); // U_{c}^T.Sigma_{c}^-1.U_{c}
    // Finally, add N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c} to m_tmp_ruru
    m_tmp_ruru += output * gmm_stats.n(c);
  }
  // Computes the inverse
  bob::math::inv(m_tmp_ruru, output);
}


void bob::learn::em::FABase::computeFn_x(const bob::learn::em::GMMStats& gmm_stats,
  blitz::Array<double,1>& output) const
{
  // Compute Fn_x = sum_{sessions h}(N*(o - m) (Normalised first order statistics)
  blitz::Range rall = blitz::Range::all();
  const size_t dim_c = getNGaussians();
  const size_t dim_d = getNInputs();
  for(size_t c=0; c<dim_c; ++c) {
    blitz::Range rc(c*dim_d,(c+1)*dim_d-1);
    blitz::Array<double,1> Fn_x_c = output(rc);
    blitz::Array<double,1> mean_c = m_cache_mean(rc);
    Fn_x_c = gmm_stats.sumPx(c,rall) - mean_c*gmm_stats.n(c);
  }
}

void bob::learn::em::FABase::estimateX(const blitz::Array<double,2>& IdPlusUSProdInv,
  const blitz::Array<double,1>& Fn_x, blitz::Array<double,1>& x) const
{
  // m_tmp_ru = UtSigmaInv * Fn_x = Ut*diag(sigma)^-1 * N*(o - m)
  bob::math::prod(m_cache_UtSigmaInv, Fn_x, m_tmp_ru);
  // x = IdPlusUSProdInv * m_cache_UtSigmaInv * Fn_x
  bob::math::prod(IdPlusUSProdInv, m_tmp_ru, x);
}


void bob::learn::em::FABase::estimateX(const bob::learn::em::GMMStats& gmm_stats, blitz::Array<double,1>& x) const
{
  if (!m_ubm) throw std::runtime_error("No UBM was set in the JFA machine.");
  computeIdPlusUSProdInv(gmm_stats, m_tmp_IdPlusUSProdInv); // Computes first term
  computeFn_x(gmm_stats, m_tmp_Fn_x); // Computes last term
  estimateX(m_tmp_IdPlusUSProdInv, m_tmp_Fn_x, x); // Estimates the value of x
}


/**
 * @date Sun Mar 31 20:15:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */


#include <bob.learn.em/IVectorTrainer.h>
#include <bob.core/check.h>
#include <bob.core/array_copy.h>
#include <bob.core/array_random.h>
#include <bob.math/inv.h>
#include <bob.core/check.h>
#include <bob.core/array_repmat.h>
#include <algorithm>

#include <bob.math/linear.h>
#include <bob.math/linsolve.h>

bob::learn::em::IVectorTrainer::IVectorTrainer(const bool update_sigma):
  m_update_sigma(update_sigma),
  m_rng(new boost::mt19937())
{}

bob::learn::em::IVectorTrainer::IVectorTrainer(const bob::learn::em::IVectorTrainer& other):
  m_update_sigma(other.m_update_sigma)
{
  m_rng                   = other.m_rng;
  m_acc_Nij_wij2.reference(bob::core::array::ccopy(other.m_acc_Nij_wij2));
  m_acc_Fnormij_wij.reference(bob::core::array::ccopy(other.m_acc_Fnormij_wij));
  m_acc_Nij.reference(bob::core::array::ccopy(other.m_acc_Nij));
  m_acc_Snormij.reference(bob::core::array::ccopy(other.m_acc_Snormij));

  m_tmp_wij.reference(bob::core::array::ccopy(other.m_tmp_wij));
  m_tmp_wij2.reference(bob::core::array::ccopy(other.m_tmp_wij2));
  m_tmp_d1.reference(bob::core::array::ccopy(other.m_tmp_d1));
  m_tmp_t1.reference(bob::core::array::ccopy(other.m_tmp_t1));
  m_tmp_dd1.reference(bob::core::array::ccopy(other.m_tmp_dd1));
  m_tmp_dt1.reference(bob::core::array::ccopy(other.m_tmp_dt1));
  m_tmp_tt1.reference(bob::core::array::ccopy(other.m_tmp_tt1));
  m_tmp_tt2.reference(bob::core::array::ccopy(other.m_tmp_tt2));
}

bob::learn::em::IVectorTrainer::~IVectorTrainer()
{
}

void bob::learn::em::IVectorTrainer::initialize(
  bob::learn::em::IVectorMachine& machine)
{
  // Initializes \f$T\f$ and \f$\Sigma\f$ of the machine
  blitz::Array<double,2>& T = machine.updateT();
  bob::core::array::randn(*m_rng, T);
  blitz::Array<double,1>& sigma = machine.updateSigma();
  sigma = machine.getUbm()->getVarianceSupervector();
  machine.precompute();
}

void bob::learn::em::IVectorTrainer::resetAccumulators(const bob::learn::em::IVectorMachine& machine)
{
   // Resize the accumulator
  const int C = machine.getNGaussians();
  const int D = machine.getNInputs();
  const int Rt = machine.getDimRt();

  // Cache
  m_acc_Nij_wij2.resize(C,Rt,Rt);
  m_acc_Fnormij_wij.resize(C,D,Rt);
  if (m_update_sigma)
  {
    m_acc_Nij.resize(C);
    m_acc_Snormij.resize(C,D);
  }

  // Tmp
  m_tmp_wij.resize(Rt);
  m_tmp_wij2.resize(Rt,Rt);
  m_tmp_d1.resize(D);
  m_tmp_t1.resize(Rt);

  m_tmp_dt1.resize(D,Rt);
  m_tmp_tt1.resize(Rt,Rt);
  m_tmp_tt2.resize(Rt,Rt);
  if (m_update_sigma)
    m_tmp_dd1.resize(D,D);

  // initialize with 0
  m_acc_Nij_wij2 = 0.;
  m_acc_Fnormij_wij = 0.;
  if (m_update_sigma)
  {
    m_acc_Nij = 0.;
    m_acc_Snormij = 0.;
  }
}


void bob::learn::em::IVectorTrainer::eStep(
  bob::learn::em::IVectorMachine& machine,
  const std::vector<bob::learn::em::GMMStats>& data)
{
  blitz::Range rall = blitz::Range::all();
  const int C = machine.getNGaussians();

  // Reinitializes accumulators to 0
  resetAccumulators(machine);

  for (std::vector<bob::learn::em::GMMStats>::const_iterator it = data.begin();
       it != data.end(); ++it)
  {
    // Computes E{wij} and E{wij.wij^{T}}
    // a. Computes \f$T^{T} \Sigma^{-1} F_{norm}\f$
    machine.computeTtSigmaInvFnorm(*it, m_tmp_t1);
    // b. Computes \f$Id + T^{T} \Sigma^{-1} T\f$
    machine.computeIdTtSigmaInvT(*it, m_tmp_tt1);
    // c. Computes \f$(Id + T^{T} \Sigma^{-1} T)^{-1}\f$

    bob::math::inv(m_tmp_tt1, m_tmp_tt2);
    // d. Computes \f$E{wij} = (Id + T^{T} \Sigma^{-1} T)^{-1} T^{T} \Sigma^{-1} F_{norm}\f$
    bob::math::prod(m_tmp_tt2, m_tmp_t1, m_tmp_wij); // E{wij}
    // e.  Computes \f$E{wij}.E{wij^{T}}\f$
    bob::math::prod(m_tmp_wij, m_tmp_wij, m_tmp_wij2);
    // f. Computes \f$E{wij.wij^{T}} = (Id + T^{T} \Sigma^{-1} T)^{-1} + E{wij}.E{wij^{T}}\f$
    m_tmp_wij2 += m_tmp_tt2; // E{wij.wij^{T}}

    if (m_update_sigma)
      m_acc_Nij += (*it).n;

    for (int c=0; c<C; ++c)
    {
      blitz::Array<double,2> acc_Nij_wij2_c = m_acc_Nij_wij2(c,rall,rall);
      blitz::Array<double,2> acc_Fnormij_wij = m_acc_Fnormij_wij(c,rall,rall);
      // acc_Nij_wij2_c += Nijc . E{wij.wij^{T}}
      acc_Nij_wij2_c += (*it).n(c) * m_tmp_wij2;
      blitz::Array<double,1> mc = machine.getUbm()->getGaussian(c)->getMean();
      // m_tmp_d1 = Fijc - Nijc * ubmmean_{c}
      m_tmp_d1 = (*it).sumPx(c,rall) - (*it).n(c)*mc; // Fnorm_c
      // m_tmp_dt1 = (Fijc - Nijc * ubmmean_{c}).E{wij}^{T}
      bob::math::prod(m_tmp_d1, m_tmp_wij, m_tmp_dt1);
      // acc_Fnormij_wij += (Fijc - Nijc * ubmmean_{c}).E{wij}^{T}
      acc_Fnormij_wij += m_tmp_dt1;
      if (m_update_sigma)
      {
        blitz::Array<double,1> acc_Snormij_c = m_acc_Snormij(c,rall);
        acc_Snormij_c += (*it).sumPxx(c,rall) - mc*((*it).sumPx(c,rall) + m_tmp_d1);
      }
    }
  }
}

void bob::learn::em::IVectorTrainer::mStep(
  bob::learn::em::IVectorMachine& machine)
{
  blitz::Range rall = blitz::Range::all();
  blitz::Array<double,2>& T = machine.updateT();
  blitz::Array<double,1>& sigma = machine.updateSigma();
  const int C = (int)machine.getNGaussians();
  const int D = (int)machine.getNInputs();
  for (int c=0; c<C; ++c)
  {
    // Solves linear system A.T = B to update T, based on accumulators of
    // the eStep()
    blitz::Array<double,2> acc_Nij_wij2_c = m_acc_Nij_wij2(c,rall,rall);
    blitz::Array<double,2> tacc_Nij_wij2_c = acc_Nij_wij2_c.transpose(1,0);
    blitz::Array<double,2> acc_Fnormij_wij_c = m_acc_Fnormij_wij(c,rall,rall);
    blitz::Array<double,2> tacc_Fnormij_wij_c = acc_Fnormij_wij_c.transpose(1,0);
    blitz::Array<double,2> T_c = T(blitz::Range(c*D,(c+1)*D-1),rall);
    blitz::Array<double,2> Tt_c = T_c.transpose(1,0);
    if (blitz::all(acc_Nij_wij2_c == 0)) // TODO
      Tt_c = 0;
    else
      bob::math::linsolve(tacc_Nij_wij2_c, Tt_c, tacc_Fnormij_wij_c);
    if (m_update_sigma)
    {
      blitz::Array<double,1> sigma_c = sigma(blitz::Range(c*D,(c+1)*D-1));
      bob::math::prod(acc_Fnormij_wij_c, Tt_c, m_tmp_dd1);
      bob::math::diag(m_tmp_dd1, m_tmp_d1);
      sigma_c = (m_acc_Snormij(c,rall) - m_tmp_d1) / m_acc_Nij(c);
    }
  }
  machine.precompute();
}


bob::learn::em::IVectorTrainer& bob::learn::em::IVectorTrainer::operator=
  (const bob::learn::em::IVectorTrainer &other)
{
  if (this != &other)
  {
    m_update_sigma = other.m_update_sigma;

    m_acc_Nij_wij2.reference(bob::core::array::ccopy(other.m_acc_Nij_wij2));
    m_acc_Fnormij_wij.reference(bob::core::array::ccopy(other.m_acc_Fnormij_wij));
    m_acc_Nij.reference(bob::core::array::ccopy(other.m_acc_Nij));
    m_acc_Snormij.reference(bob::core::array::ccopy(other.m_acc_Snormij));

    m_tmp_wij.reference(bob::core::array::ccopy(other.m_tmp_wij));
    m_tmp_wij2.reference(bob::core::array::ccopy(other.m_tmp_wij2));
    m_tmp_d1.reference(bob::core::array::ccopy(other.m_tmp_d1));
    m_tmp_t1.reference(bob::core::array::ccopy(other.m_tmp_t1));
    m_tmp_dd1.reference(bob::core::array::ccopy(other.m_tmp_dd1));
    m_tmp_dt1.reference(bob::core::array::ccopy(other.m_tmp_dt1));
    m_tmp_tt1.reference(bob::core::array::ccopy(other.m_tmp_tt1));
    m_tmp_tt2.reference(bob::core::array::ccopy(other.m_tmp_tt2));
  }
  return *this;
}

bool bob::learn::em::IVectorTrainer::operator==
  (const bob::learn::em::IVectorTrainer &other) const
{
  return m_update_sigma == other.m_update_sigma &&
         bob::core::array::isEqual(m_acc_Nij_wij2, other.m_acc_Nij_wij2) &&
         bob::core::array::isEqual(m_acc_Fnormij_wij, other.m_acc_Fnormij_wij) &&
         bob::core::array::isEqual(m_acc_Nij, other.m_acc_Nij) &&
         bob::core::array::isEqual(m_acc_Snormij, other.m_acc_Snormij);
}

bool bob::learn::em::IVectorTrainer::operator!=
  (const bob::learn::em::IVectorTrainer &other) const
{
  return !(this->operator==(other));
}

bool bob::learn::em::IVectorTrainer::is_similar_to
  (const bob::learn::em::IVectorTrainer &other, const double r_epsilon,
   const double a_epsilon) const
{
  return m_update_sigma == other.m_update_sigma &&
         bob::core::array::isClose(m_acc_Nij_wij2, other.m_acc_Nij_wij2, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_acc_Fnormij_wij, other.m_acc_Fnormij_wij, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_acc_Nij, other.m_acc_Nij, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_acc_Snormij, other.m_acc_Snormij, r_epsilon, a_epsilon);
}

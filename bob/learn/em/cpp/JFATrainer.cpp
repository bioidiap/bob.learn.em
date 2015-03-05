/**
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Joint Factor Analysis Trainer
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.learn.em/JFATrainer.h>
#include <bob.core/check.h>
#include <bob.core/array_copy.h>
#include <bob.core/array_random.h>
#include <bob.math/inv.h>
#include <bob.math/linear.h>
#include <bob.core/check.h>
#include <bob.core/array_repmat.h>
#include <algorithm>


//////////////////////////// JFATrainer ///////////////////////////
bob::learn::em::JFATrainer::JFATrainer():
  m_rng(new boost::mt19937())
{}

bob::learn::em::JFATrainer::JFATrainer(const bob::learn::em::JFATrainer& other):
 m_rng(other.m_rng)
{}

bob::learn::em::JFATrainer::~JFATrainer()
{}

bob::learn::em::JFATrainer& bob::learn::em::JFATrainer::operator=
(const bob::learn::em::JFATrainer& other)
{
  if (this != &other)
  {
    //m_max_iterations = other.m_max_iterations;
    m_rng = other.m_rng;
  }
  return *this;
}

bool bob::learn::em::JFATrainer::operator==(const bob::learn::em::JFATrainer& b) const
{
  //return m_max_iterations == b.m_max_iterations && *m_rng == *(b.m_rng);
  return *m_rng == *(b.m_rng);
}

bool bob::learn::em::JFATrainer::operator!=(const bob::learn::em::JFATrainer& b) const
{
  return !(this->operator==(b));
}

bool bob::learn::em::JFATrainer::is_similar_to(const bob::learn::em::JFATrainer& b,
  const double r_epsilon, const double a_epsilon) const
{
  //return m_max_iterations == b.m_max_iterations && *m_rng == *(b.m_rng);
  return *m_rng == *(b.m_rng);
}

void bob::learn::em::JFATrainer::initialize(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  m_base_trainer.initUbmNidSumStatistics(machine.getBase(), ar);
  m_base_trainer.initializeXYZ(ar);

  blitz::Array<double,2>& U = machine.updateU();
  bob::core::array::randn(*m_rng, U);
  blitz::Array<double,2>& V = machine.updateV();
  bob::core::array::randn(*m_rng, V);
  blitz::Array<double,1>& D = machine.updateD();
  bob::core::array::randn(*m_rng, D);
  machine.precompute();
}

void bob::learn::em::JFATrainer::eStep1(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  const bob::learn::em::FABase& base = machine.getBase();
  m_base_trainer.updateY(base, ar);
  m_base_trainer.computeAccumulatorsV(base, ar);
}

void bob::learn::em::JFATrainer::mStep1(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  blitz::Array<double,2>& V = machine.updateV();
  m_base_trainer.updateV(V);
}

void bob::learn::em::JFATrainer::finalize1(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  const bob::learn::em::FABase& base = machine.getBase();
  m_base_trainer.updateY(base, ar);
}


void bob::learn::em::JFATrainer::eStep2(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  const bob::learn::em::FABase& base = machine.getBase();
  m_base_trainer.updateX(base, ar);
  m_base_trainer.computeAccumulatorsU(base, ar);
}

void bob::learn::em::JFATrainer::mStep2(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  blitz::Array<double,2>& U = machine.updateU();
  m_base_trainer.updateU(U);
  machine.precompute();
}

void bob::learn::em::JFATrainer::finalize2(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  const bob::learn::em::FABase& base = machine.getBase();
  m_base_trainer.updateX(base, ar);
}


void bob::learn::em::JFATrainer::eStep3(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  const bob::learn::em::FABase& base = machine.getBase();
  m_base_trainer.updateZ(base, ar);
  m_base_trainer.computeAccumulatorsD(base, ar);
}

void bob::learn::em::JFATrainer::mStep3(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  blitz::Array<double,1>& d = machine.updateD();
  m_base_trainer.updateD(d);
}

void bob::learn::em::JFATrainer::finalize3(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
}

/*
void bob::learn::em::JFATrainer::train_loop(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  // V subspace
  for (size_t i=0; i<m_max_iterations; ++i) {
    eStep1(machine, ar);
    mStep1(machine, ar);
  }
  finalize1(machine, ar);
  // U subspace
  for (size_t i=0; i<m_max_iterations; ++i) {
    eStep2(machine, ar);
    mStep2(machine, ar);
  }
  finalize2(machine, ar);
  // d subspace
  for (size_t i=0; i<m_max_iterations; ++i) {
    eStep3(machine, ar);
    mStep3(machine, ar);
  }
  finalize3(machine, ar);
}*/

/*
void bob::learn::em::JFATrainer::train(bob::learn::em::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar)
{
  initialize(machine, ar);
  train_loop(machine, ar);
}
*/

void bob::learn::em::JFATrainer::enroll(bob::learn::em::JFAMachine& machine,
  const std::vector<boost::shared_ptr<bob::learn::em::GMMStats> >& ar,
  const size_t n_iter)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > > vvec;
  vvec.push_back(ar);

  const bob::learn::em::FABase& fb = machine.getJFABase()->getBase();

  m_base_trainer.initUbmNidSumStatistics(fb, vvec);
  m_base_trainer.initializeXYZ(vvec);

  for (size_t i=0; i<n_iter; ++i) {
    m_base_trainer.updateY(fb, vvec);
    m_base_trainer.updateX(fb, vvec);
    m_base_trainer.updateZ(fb, vvec);
  }

  const blitz::Array<double,1> y(m_base_trainer.getY()[0]);
  const blitz::Array<double,1> z(m_base_trainer.getZ()[0]);
  machine.setY(y);
  machine.setZ(z);
}

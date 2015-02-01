/**
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Joint Factor Analysis Trainer
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.learn.misc/ISVTrainer.h>
#include <bob.core/check.h>
#include <bob.core/array_copy.h>
#include <bob.core/array_random.h>
#include <bob.math/inv.h>
#include <bob.math/linear.h>
#include <bob.core/check.h>
#include <bob.core/array_repmat.h>
#include <algorithm>


//////////////////////////// ISVTrainer ///////////////////////////
bob::learn::misc::ISVTrainer::ISVTrainer(const size_t max_iterations, const double relevance_factor):
  EMTrainer<bob::learn::misc::ISVBase, std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > >
    (0.001, max_iterations, false),
  m_relevance_factor(relevance_factor)
{
}

bob::learn::misc::ISVTrainer::ISVTrainer(const bob::learn::misc::ISVTrainer& other):
  EMTrainer<bob::learn::misc::ISVBase, std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > >
    (other.m_convergence_threshold, other.m_max_iterations,
     other.m_compute_likelihood),
  m_relevance_factor(other.m_relevance_factor)
{
}

bob::learn::misc::ISVTrainer::~ISVTrainer()
{
}

bob::learn::misc::ISVTrainer& bob::learn::misc::ISVTrainer::operator=
(const bob::learn::misc::ISVTrainer& other)
{
  if (this != &other)
  {
    bob::learn::misc::EMTrainer<bob::learn::misc::ISVBase,
      std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > >::operator=(other);
    m_relevance_factor = other.m_relevance_factor;
  }
  return *this;
}

bool bob::learn::misc::ISVTrainer::operator==(const bob::learn::misc::ISVTrainer& b) const
{
  return bob::learn::misc::EMTrainer<bob::learn::misc::ISVBase,
            std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > >::operator==(b) &&
          m_relevance_factor == b.m_relevance_factor;
}

bool bob::learn::misc::ISVTrainer::operator!=(const bob::learn::misc::ISVTrainer& b) const
{
  return !(this->operator==(b));
}

bool bob::learn::misc::ISVTrainer::is_similar_to(const bob::learn::misc::ISVTrainer& b,
  const double r_epsilon, const double a_epsilon) const
{
  return bob::learn::misc::EMTrainer<bob::learn::misc::ISVBase,
            std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > >::is_similar_to(b, r_epsilon, a_epsilon) &&
          m_relevance_factor == b.m_relevance_factor;
}

void bob::learn::misc::ISVTrainer::initialize(bob::learn::misc::ISVBase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > >& ar)
{
  m_base_trainer.initUbmNidSumStatistics(machine.getBase(), ar);
  m_base_trainer.initializeXYZ(ar);

  blitz::Array<double,2>& U = machine.updateU();
  bob::core::array::randn(*m_rng, U);
  initializeD(machine);
  machine.precompute();
}

void bob::learn::misc::ISVTrainer::initializeD(bob::learn::misc::ISVBase& machine) const
{
  // D = sqrt(variance(UBM) / relevance_factor)
  blitz::Array<double,1> d = machine.updateD();
  d = sqrt(machine.getBase().getUbmVariance() / m_relevance_factor);
}

void bob::learn::misc::ISVTrainer::finalize(bob::learn::misc::ISVBase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > >& ar)
{
}

void bob::learn::misc::ISVTrainer::eStep(bob::learn::misc::ISVBase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > >& ar)
{
  m_base_trainer.resetXYZ();

  const bob::learn::misc::FABase& base = machine.getBase();
  m_base_trainer.updateX(base, ar);
  m_base_trainer.updateZ(base, ar);
  m_base_trainer.computeAccumulatorsU(base, ar);
}

void bob::learn::misc::ISVTrainer::mStep(bob::learn::misc::ISVBase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > >& ar)
{
  blitz::Array<double,2>& U = machine.updateU();
  m_base_trainer.updateU(U);
  machine.precompute();
}

double bob::learn::misc::ISVTrainer::computeLikelihood(bob::learn::misc::ISVBase& machine)
{
  // TODO
  return 0;
}

void bob::learn::misc::ISVTrainer::enrol(bob::learn::misc::ISVMachine& machine,
  const std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> >& ar,
  const size_t n_iter)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > vvec;
  vvec.push_back(ar);

  const bob::learn::misc::FABase& fb = machine.getISVBase()->getBase();

  m_base_trainer.initUbmNidSumStatistics(fb, vvec);
  m_base_trainer.initializeXYZ(vvec);

  for (size_t i=0; i<n_iter; ++i) {
    m_base_trainer.updateX(fb, vvec);
    m_base_trainer.updateZ(fb, vvec);
  }

  const blitz::Array<double,1> z(m_base_trainer.getZ()[0]);
  machine.setZ(z);
}




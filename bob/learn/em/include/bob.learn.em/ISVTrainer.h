/**
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief JFA functions
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_ISVTRAINER_H
#define BOB_LEARN_EM_ISVTRAINER_H

#include <blitz/array.h>
#include <bob.learn.em/GMMStats.h>
#include <bob.learn.em/FABaseTrainer.h>
#include <bob.learn.em/ISVMachine.h>
#include <vector>

#include <map>
#include <string>
#include <bob.core/array_copy.h>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <bob.core/logging.h>

namespace bob { namespace learn { namespace em {

class ISVTrainer
{
  public:
    /**
     * @brief Constructor
     */
    ISVTrainer(const double relevance_factor=4.);

    /**
     * @brief Copy onstructor
     */
    ISVTrainer(const ISVTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~ISVTrainer();

    /**
     * @brief Assignment operator
     */
    ISVTrainer& operator=(const ISVTrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const ISVTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const ISVTrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const ISVTrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief This methods performs some initialization before the EM loop.
     */
    virtual void initialize(bob::learn::em::ISVBase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);

    /**
     * @brief Calculates and saves statistics across the dataset
     * The statistics will be used in the mStep() that follows.
     */
    virtual void eStep(bob::learn::em::ISVBase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);

    /**
     * @brief Performs a maximization step to update the parameters of the
     * factor analysis model.
     */
    virtual void mStep(bob::learn::em::ISVBase& machine);

    /**
     * @brief Computes the average log likelihood using the current estimates
     * of the latent variables.
     */
    virtual double computeLikelihood(bob::learn::em::ISVBase& machine);

    /**
     * @brief Enrol a client
     */
    void enroll(bob::learn::em::ISVMachine& machine,
      const std::vector<boost::shared_ptr<bob::learn::em::GMMStats> >& features,
      const size_t n_iter);

    /**
     * @brief Get the x speaker factors
     */
    const std::vector<blitz::Array<double,2> >& getX() const
    { return m_base_trainer.getX(); }
    /**
     * @brief Get the z speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getZ() const
    { return m_base_trainer.getZ(); }
    /**
     * @brief Set the x speaker factors
     */
    void setX(const std::vector<blitz::Array<double,2> >& X)
    { m_base_trainer.setX(X); }
    /**
     * @brief Set the z speaker factors
     */
    void setZ(const std::vector<blitz::Array<double,1> >& z)
    { m_base_trainer.setZ(z); }

    /**
     * @brief Getters for the accumulators
     */
    const blitz::Array<double,3>& getAccUA1() const
    { return m_base_trainer.getAccUA1(); }
    const blitz::Array<double,2>& getAccUA2() const
    { return m_base_trainer.getAccUA2(); }

    /**
     * @brief Setters for the accumulators, Very useful if the e-Step needs
     * to be parallelized.
     */
    void setAccUA1(const blitz::Array<double,3>& acc)
    { m_base_trainer.setAccUA1(acc); }
    void setAccUA2(const blitz::Array<double,2>& acc)
    { m_base_trainer.setAccUA2(acc); }

    /**
     * @brief Sets the Random Number Generator
     */
    void setRng(const boost::shared_ptr<boost::mt19937> rng)
    { m_rng = rng; }

    /**
     * @brief Gets the Random Number Generator
     */
    const boost::shared_ptr<boost::mt19937> getRng() const
    { return m_rng; }


  private:
    /**
     * @brief Initialize D to sqrt(ubm_var/relevance_factor)
     */
    void initializeD(bob::learn::em::ISVBase& machine) const;

    // Attributes
    bob::learn::em::FABaseTrainer m_base_trainer;

    double m_relevance_factor;

    boost::shared_ptr<boost::mt19937> m_rng; ///< The random number generator for the inialization};
};

} } } // namespaces

#endif /* BOB_LEARN_EM_ISVTRAINER_H */

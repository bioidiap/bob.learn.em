/**
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief JFA functions
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_JFATRAINER_H
#define BOB_LEARN_EM_JFATRAINER_H

#include <blitz/array.h>
#include <bob.learn.em/GMMStats.h>
#include <bob.learn.em/FABaseTrainer.h>
#include <bob.learn.em/JFAMachine.h>
#include <vector>

#include <map>
#include <string>
#include <bob.core/array_copy.h>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <bob.core/logging.h>

namespace bob { namespace learn { namespace em {

class JFATrainer
{
  public:
    /**
     * @brief Constructor
     */
    JFATrainer();

    /**
     * @brief Copy onstructor
     */
    JFATrainer(const JFATrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~JFATrainer();

    /**
     * @brief Assignment operator
     */
    JFATrainer& operator=(const JFATrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const JFATrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const JFATrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const JFATrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Sets the maximum number of EM-like iterations (for each subspace)
     */
    //void setMaxIterations(const size_t max_iterations)
    //{ m_max_iterations = max_iterations; }

    /**
     * @brief Gets the maximum number of EM-like iterations (for each subspace)
     */
    //size_t getMaxIterations() const
    //{ return m_max_iterations; }

    /**
     * @brief This methods performs some initialization before the EM loop.
     */
    virtual void initialize(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);

    /**
     * @brief This methods performs the e-Step to train the first subspace V
     */
    virtual void eStep1(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods performs the m-Step to train the first subspace V
     */
    virtual void mStep1(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods performs the finalization after training the first
     * subspace V
     */
    virtual void finalize1(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods performs the e-Step to train the second subspace U
     */
    virtual void eStep2(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods performs the m-Step to train the second subspace U
     */
    virtual void mStep2(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods performs the finalization after training the second
     * subspace U
     */
    virtual void finalize2(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods performs the e-Step to train the third subspace d
     */
    virtual void eStep3(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods performs the m-Step to train the third subspace d
     */
    virtual void mStep3(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods performs the finalization after training the third
     * subspace d
     */
    virtual void finalize3(bob::learn::em::JFABase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);

    /**
     * @brief This methods performs the main loops to train the subspaces U, V and d
     */
    //virtual void train_loop(bob::learn::em::JFABase& machine,
      //const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);
    /**
     * @brief This methods trains the subspaces U, V and d
     */
    //virtual void train(bob::learn::em::JFABase& machine,
      //const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& ar);

    /**
     * @brief Enrol a client
     */
    void enroll(bob::learn::em::JFAMachine& machine,
      const std::vector<boost::shared_ptr<bob::learn::em::GMMStats> >& features,
      const size_t n_iter);

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

    /**
     * @brief Get the x speaker factors
     */
    const std::vector<blitz::Array<double,2> >& getX() const
    { return m_base_trainer.getX(); }
    /**
     * @brief Get the y speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getY() const
    { return m_base_trainer.getY(); }
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
     * @brief Set the y speaker factors
     */
    void setY(const std::vector<blitz::Array<double,1> >& y)
    { m_base_trainer.setY(y); }
    /**
     * @brief Set the z speaker factors
     */
    void setZ(const std::vector<blitz::Array<double,1> >& z)
    { m_base_trainer.setZ(z); }

    /**
     * @brief Getters for the accumulators
     */
    const blitz::Array<double,3>& getAccVA1() const
    { return m_base_trainer.getAccVA1(); }
    const blitz::Array<double,2>& getAccVA2() const
    { return m_base_trainer.getAccVA2(); }
    const blitz::Array<double,3>& getAccUA1() const
    { return m_base_trainer.getAccUA1(); }
    const blitz::Array<double,2>& getAccUA2() const
    { return m_base_trainer.getAccUA2(); }
    const blitz::Array<double,1>& getAccDA1() const
    { return m_base_trainer.getAccDA1(); }
    const blitz::Array<double,1>& getAccDA2() const
    { return m_base_trainer.getAccDA2(); }

    /**
     * @brief Setters for the accumulators, Very useful if the e-Step needs
     * to be parallelized.
     */
    void setAccVA1(const blitz::Array<double,3>& acc)
    { m_base_trainer.setAccVA1(acc); }
    void setAccVA2(const blitz::Array<double,2>& acc)
    { m_base_trainer.setAccVA2(acc); }
    void setAccUA1(const blitz::Array<double,3>& acc)
    { m_base_trainer.setAccUA1(acc); }
    void setAccUA2(const blitz::Array<double,2>& acc)
    { m_base_trainer.setAccUA2(acc); }
    void setAccDA1(const blitz::Array<double,1>& acc)
    { m_base_trainer.setAccDA1(acc); }
    void setAccDA2(const blitz::Array<double,1>& acc)
    { m_base_trainer.setAccDA2(acc); }


  private:
    // Attributes
    //size_t m_max_iterations;
    boost::shared_ptr<boost::mt19937> m_rng; ///< The random number generator for the inialization
    bob::learn::em::FABaseTrainer m_base_trainer;
};

} } } // namespaces

#endif /* BOB_LEARN_EM_JFATRAINER_H */

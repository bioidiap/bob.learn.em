/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * @brief This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_MISC_ML_GMMTRAINER_H
#define BOB_LEARN_MISC_ML_GMMTRAINER_H

#include <bob.learn.misc/GMMBaseTrainer.h>
#include <limits>

namespace bob { namespace learn { namespace misc {

/**
 * @brief This class implements the maximum likelihood M-step of the
 *   expectation-maximisation algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop,
 *  "Pattern recognition and machine learning", 2006
 */
class ML_GMMTrainer{
  public:
    /**
     * @brief Default constructor
     */
    ML_GMMTrainer(boost::shared_ptr<bob::learn::misc::GMMBaseTrainer> gmm_base_trainer);

    /**
     * @brief Copy constructor
     */
    ML_GMMTrainer(const ML_GMMTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~ML_GMMTrainer();

    /**
     * @brief Initialisation before the EM steps
     */
    virtual void initialize(bob::learn::misc::GMMMachine& gmm);

    /**
     * @brief Performs a maximum likelihood (ML) update of the GMM parameters
     * using the accumulated statistics in m_ss
     * Implements EMTrainer::mStep()
     */
    virtual void mStep(bob::learn::misc::GMMMachine& gmm);

    /**
     * @brief Assigns from a different ML_GMMTrainer
     */
    ML_GMMTrainer& operator=(const ML_GMMTrainer &other);

    /**
     * @brief Equal to
     */
    bool operator==(const ML_GMMTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const ML_GMMTrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const ML_GMMTrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;
      
    
    boost::shared_ptr<bob::learn::misc::GMMBaseTrainer> getGMMBaseTrainer()
    {return m_gmm_base_trainer;}
    
    void setGMMBaseTrainer(boost::shared_ptr<bob::learn::misc::GMMBaseTrainer> gmm_base_trainer)
    {m_gmm_base_trainer = gmm_base_trainer;}
    

  protected:

    /**
    Base Trainer for the MAP algorithm. Basically implements the e-step
    */ 
    boost::shared_ptr<bob::learn::misc::GMMBaseTrainer> m_gmm_base_trainer;


  private:
    /**
     * @brief Add cache to avoid re-allocation at each iteration
     */
    mutable blitz::Array<double,1> m_cache_ss_n_thresholded;
};

} } } // namespaces

#endif // BOB_LEARN_MISC_ML_GMMTRAINER_H

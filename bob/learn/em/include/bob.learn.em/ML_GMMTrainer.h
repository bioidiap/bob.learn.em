/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * @brief This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine.
 * @details See Section 9.2.2 of Bishop, "Pattern recognition and machine learning", 2006
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_ML_GMMTRAINER_H
#define BOB_LEARN_EM_ML_GMMTRAINER_H

#include <bob.learn.em/GMMBaseTrainer.h>
#include <limits>

namespace bob { namespace learn { namespace em {

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
    ML_GMMTrainer(const bool update_means=true,
                  const bool update_variances=false,
                  const bool update_weights=false,
                  const double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon());

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
    void initialize(bob::learn::em::GMMMachine& gmm);

    /**
     * @brief Calculates and saves statistics across the dataset,
     * and saves these as m_ss. Calculates the average
     * log likelihood of the observations given the GMM,
     * and returns this in average_log_likelihood.
     *
     * The statistics, m_ss, will be used in the mStep() that follows.
     * Implements EMTrainer::eStep(double &)
     */
     void eStep(bob::learn::em::GMMMachine& gmm,
      const blitz::Array<double,2>& data){
      m_gmm_base_trainer.eStep(gmm,data);
     }

    /**
     * @brief Performs a maximum likelihood (ML) update of the GMM parameters
     * using the accumulated statistics in m_ss
     * Implements EMTrainer::mStep()
     */
    void mStep(bob::learn::em::GMMMachine& gmm);

    /**
     * @brief Computes the likelihood using current estimates of the latent
     * variables
     */
    double computeLikelihood(bob::learn::em::GMMMachine& gmm){
      return m_gmm_base_trainer.computeLikelihood(gmm);
    }


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


    bob::learn::em::GMMBaseTrainer& base_trainer(){return m_gmm_base_trainer;}

  protected:

    /**
    Base Trainer for the MAP algorithm. Basically implements the e-step
    */
    bob::learn::em::GMMBaseTrainer m_gmm_base_trainer;


  private:
    /**
     * @brief Add cache to avoid re-allocation at each iteration
     */
    mutable blitz::Array<double,1> m_cache_ss_n_thresholded;
};

} } } // namespaces

#endif // BOB_LEARN_EM_ML_GMMTRAINER_H

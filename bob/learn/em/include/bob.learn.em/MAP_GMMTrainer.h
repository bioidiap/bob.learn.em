/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * @brief This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation.
 * @details See Section 3.4 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000. We use a "single adaptation coefficient", alpha_i, and thus a single relevance factor, r.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_MAP_GMMTRAINER_H
#define BOB_LEARN_EM_MAP_GMMTRAINER_H

#include <bob.learn.em/GMMBaseTrainer.h>
#include <limits>

namespace bob { namespace learn { namespace em {

/**
 * @brief This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation.
 * @details See Section 3.4 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000. We use a "single adaptation coefficient", alpha_i, and thus a single relevance factor, r.
 */
class MAP_GMMTrainer
{
  public:
    /**
     * @brief Default constructor
     */
    MAP_GMMTrainer(
      const bool update_means=true,
      const bool update_variances=false,
      const bool update_weights=false,
      const double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon(),
      const bool reynolds_adaptation=false,
      const double relevance_factor=4,
      const double alpha=0.5,
      boost::shared_ptr<bob::learn::em::GMMMachine> prior_gmm = boost::shared_ptr<bob::learn::em::GMMMachine>());

    /**
     * @brief Copy constructor
     */
    MAP_GMMTrainer(const MAP_GMMTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~MAP_GMMTrainer();

    /**
     * @brief Initialization
     */
    void initialize(bob::learn::em::GMMMachine& gmm);

    /**
     * @brief Assigns from a different MAP_GMMTrainer
     */
    MAP_GMMTrainer& operator=(const MAP_GMMTrainer &other);

    /**
     * @brief Equal to
     */
    bool operator==(const MAP_GMMTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const MAP_GMMTrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const MAP_GMMTrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Set the GMM to use as a prior for MAP adaptation.
     * Generally, this is a "universal background model" (UBM),
     * also referred to as a "world model".
     */
    bool setPriorGMM(boost::shared_ptr<bob::learn::em::GMMMachine> prior_gmm);

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
     * @brief Performs a maximum a posteriori (MAP) update of the GMM
     * parameters using the accumulated statistics in m_ss and the
     * parameters of the prior model
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

    bool getReynoldsAdaptation()
    {return m_reynolds_adaptation;}

    void setReynoldsAdaptation(const bool reynolds_adaptation)
    {m_reynolds_adaptation = reynolds_adaptation;}


    double getRelevanceFactor()
    {return m_relevance_factor;}

    void setRelevanceFactor(const double relevance_factor)
    {m_relevance_factor = relevance_factor;}


    double getAlpha()
    {return m_alpha;}

    void setAlpha(const double alpha)
    {m_alpha = alpha;}

    bob::learn::em::GMMBaseTrainer& base_trainer(){return m_gmm_base_trainer;}


  protected:

    /**
     * The relevance factor for MAP adaptation, r (see Reynolds et al., \"Speaker Verification Using Adapted Gaussian Mixture Models\", Digital Signal Processing, 2000).
     */
    double m_relevance_factor;

    /**
    Base Trainer for the MAP algorithm. Basically implements the e-step
    */
    bob::learn::em::GMMBaseTrainer m_gmm_base_trainer;

    /**
     * The GMM to use as a prior for MAP adaptation.
     * Generally, this is a "universal background model" (UBM),
     * also referred to as a "world model"
     */
    boost::shared_ptr<bob::learn::em::GMMMachine> m_prior_gmm;

    /**
     * The alpha for the Torch3-like adaptation
     */
    double m_alpha;
    /**
     * Whether Torch3-like adaptation should be used or not
     */
    bool m_reynolds_adaptation;

  private:
    /// cache to avoid re-allocation
    mutable blitz::Array<double,1> m_cache_alpha;
    mutable blitz::Array<double,1> m_cache_ml_weights;
};

} } } // namespaces

#endif // BOB_LEARN_EM_MAP_GMMTRAINER_H

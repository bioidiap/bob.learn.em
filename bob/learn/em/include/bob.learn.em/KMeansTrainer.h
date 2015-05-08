/**
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */
#ifndef BOB_LEARN_EM_KMEANSTRAINER_H
#define BOB_LEARN_EM_KMEANSTRAINER_H

#include <bob.learn.em/KMeansMachine.h>
#include <boost/version.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace bob { namespace learn { namespace em {

/**
 * Trains a KMeans machine.
 * @brief This class implements the expectation-maximisation algorithm for a k-means machine.
 * @details See Section 9.1 of Bishop, "Pattern recognition and machine learning", 2006
 *          It uses a random initialisation of the means followed by the expectation-maximization algorithm
 */
class KMeansTrainer
{
  public:
    /**
     * @brief This enumeration defines different initialization methods for
     * K-means
     */
    typedef enum {
      RANDOM=0,
      RANDOM_NO_DUPLICATE
#if BOOST_VERSION >= 104700
      ,
      KMEANS_PLUS_PLUS
#endif
    }
    InitializationMethod;

    /**
     * @brief Constructor
     */
    KMeansTrainer(InitializationMethod=RANDOM);

    /**
     * @brief Virtualize destructor
     */
    virtual ~KMeansTrainer() {}

    /**
     * @brief Copy constructor
     */
    KMeansTrainer(const KMeansTrainer& other);

    /**
     * @brief Assigns from a different machine
     */
    KMeansTrainer& operator=(const KMeansTrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const KMeansTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const KMeansTrainer& b) const;

    /**
     * @brief The name for this trainer
     */
    virtual std::string name() const { return "KMeansTrainer"; }

    /**
     * @brief Initialise the means randomly.
     * Data is split into as many chunks as there are means,
     * then each mean is set to a random example within each chunk.
     */
    void initialize(bob::learn::em::KMeansMachine& kMeansMachine,
      const blitz::Array<double,2>& sampler);

    /**
     * @brief Accumulate across the dataset:
     * - zeroeth and first order statistics
     * - average (Square Euclidean) distance from the closest mean
     * Implements EMTrainer::eStep(double &)
     */
    void eStep(bob::learn::em::KMeansMachine& kmeans,
      const blitz::Array<double,2>& data);

    /**
     * @brief Updates the mean based on the statistics from the E-step.
     */
    void mStep(bob::learn::em::KMeansMachine& kmeans);

    /**
     * @brief This functions returns the average min (Square Euclidean)
     * distance (average distance to the closest mean)
     */
    double computeLikelihood(bob::learn::em::KMeansMachine& kmeans);


    /**
     * @brief Reset the statistics accumulators
     * to the correct size and a value of zero.
     */
    void resetAccumulators(bob::learn::em::KMeansMachine& kMeansMachine);

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
     * @brief Sets the initialization method used to generate the initial means
     */
    void setInitializationMethod(InitializationMethod v) { m_initialization_method = v; }

    /**
     * @brief Gets the initialization method used to generate the initial means
     */
    InitializationMethod getInitializationMethod() const { return m_initialization_method; }

    /**
     * @brief Returns the internal statistics. Useful to parallelize the E-step
     */
    const blitz::Array<double,1>& getZeroethOrderStats() const { return m_zeroethOrderStats; }
    const blitz::Array<double,2>& getFirstOrderStats() const { return m_firstOrderStats; }
    double getAverageMinDistance() const { return m_average_min_distance; }
    /**
     * @brief Sets the internal statistics. Useful to parallelize the E-step
     */
    void setZeroethOrderStats(const blitz::Array<double,1>& zeroethOrderStats);
    void setFirstOrderStats(const blitz::Array<double,2>& firstOrderStats);
    void setAverageMinDistance(const double value) { m_average_min_distance = value; }


  private:

    /**
     * @brief The initialization method
     * Check that there is no duplicated means during the random initialization
     */
    InitializationMethod m_initialization_method;

    /**
     * @brief The random number generator for the inialization
     */
    boost::shared_ptr<boost::mt19937> m_rng;

    /**
     * @brief Average min (Square Euclidean) distance
     */
    double m_average_min_distance;

    /**
     * @brief Zeroeth order statistics accumulator.
     * The k'th value in m_zeroethOrderStats is the denominator of
     * equation 9.4, Bishop, "Pattern recognition and machine learning", 2006
     */
    blitz::Array<double,1> m_zeroethOrderStats;

    /**
     * @brief First order statistics accumulator.
     * The k'th row of m_firstOrderStats is the numerator of
     * equation 9.4, Bishop, "Pattern recognition and machine learning", 2006
     */
    blitz::Array<double,2> m_firstOrderStats;
};

} } } // namespaces

#endif // BOB_LEARN_EM_KMEANSTRAINER_H

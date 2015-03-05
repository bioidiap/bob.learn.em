/**
 * @date Tue Jan 27 16:47:00 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * @brief A base class for Joint Factor Analysis-like machines
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_JFAMACHINE_H
#define BOB_LEARN_EM_JFAMACHINE_H

#include <stdexcept>

#include <bob.learn.em/JFABase.h>
#include <bob.learn.em/GMMMachine.h>
#include <bob.learn.em/LinearScoring.h>

#include <bob.io.base/HDF5File.h>
#include <boost/shared_ptr.hpp>

namespace bob { namespace learn { namespace em {


/**
 * @brief A JFAMachine which is associated to a JFABase that contains
 *   U, V and D matrices. The JFAMachine describes the identity part
 *   (latent variables y and z)
 * TODO: add a reference to the journal articles
 */
class JFAMachine
{
  public:
    /**
     * @brief Default constructor. Builds an otherwise invalid 0 x 0 JFAMachine
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    JFAMachine();

    /**
     * @brief Constructor. Builds a new JFAMachine.
     *
     * @param jfa_base The JFABase associated with this machine
     */
    JFAMachine(const boost::shared_ptr<bob::learn::em::JFABase> jfa_base);

    /**
     * @brief Copy constructor
     */
    JFAMachine(const JFAMachine& other);

    /**
     * @deprecated Starts a new JFAMachine from an existing Configuration object.
     */
    JFAMachine(bob::io::base::HDF5File& config);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~JFAMachine();

    /**
     * @brief Assigns from a different JFA machine
     */
    JFAMachine& operator=(const JFAMachine &other);

    /**
     * @brief Equal to
     */
    bool operator==(const JFAMachine& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const JFAMachine& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const JFAMachine& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Saves machine to an HDF5 file
     */
    void save(bob::io::base::HDF5File& config) const;

    /**
     * @brief Loads data from an existing configuration object. Resets
     * the current state.
     */
    void load(bob::io::base::HDF5File& config);

    /**
     * @brief Returns the number of Gaussian components C
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getNGaussians() const
    { return m_jfa_base->getNGaussians(); }

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getNInputs() const
    { return m_jfa_base->getNInputs(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getSupervectorLength() const
    { return m_jfa_base->getSupervectorLength(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const
    { return m_jfa_base->getDimRu(); }

    /**
     * @brief Returns the size/rank rv of the V matrix
     */
    const size_t getDimRv() const
    { return m_jfa_base->getDimRv(); }

    /**
     * @brief Returns the x session factor
     */
    const blitz::Array<double,1>& getX() const
    { return m_cache_x; }

    /**
     * @brief Returns the y speaker factor
     */
    const blitz::Array<double,1>& getY() const
    { return m_y; }

    /**
     * @brief Returns the z speaker factor
     */
    const blitz::Array<double,1>& getZ() const
    { return m_z; }

    /**
     * @brief Returns the y speaker factors in order to update it
     */
    blitz::Array<double,1>& updateY()
    { return m_y; }

    /**
     * @brief Returns the z speaker factors in order to update it
     */
    blitz::Array<double,1>& updateZ()
    { return m_z; }

    /**
     * @brief Returns the y speaker factors
     */
    void setY(const blitz::Array<double,1>& y);

    /**
     * @brief Returns the V matrix
     */
    void setZ(const blitz::Array<double,1>& z);

    /**
     * @brief Returns the JFABase
     */
    const boost::shared_ptr<bob::learn::em::JFABase> getJFABase() const
    { return m_jfa_base; }

    /**
     * @brief Sets the JFABase
     */
    void setJFABase(const boost::shared_ptr<bob::learn::em::JFABase> jfa_base);


    /**
     * @brief Estimates x from the GMM statistics considering the LPT
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::learn::em::GMMStats& gmm_stats, blitz::Array<double,1>& x) const
    { m_jfa_base->estimateX(gmm_stats, x); }
    /**
     * @brief Estimates Ux from the GMM statistics considering the LPT
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateUx(const bob::learn::em::GMMStats& gmm_stats, blitz::Array<double,1>& Ux);

   /**
    * @brief Execute the machine
    *
    * @param input input data used by the machine
    * @warning Inputs are checked
    * @return score value computed by the machine
    */
    double forward(const bob::learn::em::GMMStats& input);
    /**
     * @brief Computes a score for the given UBM statistics and given the
     * Ux vector
     */
    double forward(const bob::learn::em::GMMStats& gmm_stats,
      const blitz::Array<double,1>& Ux);

    /**
     * @brief Execute the machine
     *
     * @param input input data used by the machine
     * @param score value computed by the machine
     * @warning Inputs are NOT checked
     */
    double forward_(const bob::learn::em::GMMStats& input);

  private:
    /**
     * @brief Resize latent variable according to the JFABase
     */
    void resize();
    /**
     * @brief Resize working arrays
     */
    void resizeTmp();
    /**
     * @brief Update the cache
     */
    void updateCache();

    // UBM
    boost::shared_ptr<bob::learn::em::JFABase> m_jfa_base;

    // y and z vectors/factors learned during the enrollment procedure
    blitz::Array<double,1> m_y;
    blitz::Array<double,1> m_z;

    // cache
    blitz::Array<double,1> m_cache_mVyDz;
    mutable blitz::Array<double,1> m_cache_x;

    // x vector/factor in cache when computing scores
    mutable blitz::Array<double,1> m_tmp_Ux;
};

} } } // namespaces

#endif // BOB_LEARN_EM_JFAMACHINE_H

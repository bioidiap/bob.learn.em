/**
 * @date Tue Jan 27 16:06:00 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * @brief A base class for Joint Factor Analysis-like machines
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_ISVMACHINE_H
#define BOB_LEARN_EM_ISVMACHINE_H

#include <stdexcept>

#include <bob.learn.em/ISVBase.h>
#include <bob.learn.em/GMMMachine.h>
#include <bob.learn.em/LinearScoring.h>

#include <bob.io.base/HDF5File.h>
#include <boost/shared_ptr.hpp>

namespace bob { namespace learn { namespace em {


/**
 * @brief A ISVMachine which is associated to a ISVBase that contains
 *   U D matrices.
 * TODO: add a reference to the journal articles
 */
class ISVMachine
{
  public:
    /**
     * @brief Default constructor. Builds an otherwise invalid 0 x 0 ISVMachine
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    ISVMachine();

    /**
     * @brief Constructor. Builds a new ISVMachine.
     *
     * @param isv_base The ISVBase associated with this machine
     */
    ISVMachine(const boost::shared_ptr<bob::learn::em::ISVBase> isv_base);

    /**
     * @brief Copy constructor
     */
    ISVMachine(const ISVMachine& other);

    /**
     * @brief Starts a new ISVMachine from an existing Configuration object.
     */
    ISVMachine(bob::io::base::HDF5File& config);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~ISVMachine();

    /**
     * @brief Assigns from a different ISV machine
     */
    ISVMachine& operator=(const ISVMachine &other);

    /**
     * @brief Equal to
     */
    bool operator==(const ISVMachine& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const ISVMachine& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const ISVMachine& b, const double r_epsilon=1e-5,
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
    { return m_isv_base->getNGaussians(); }

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getNInputs() const
    { return m_isv_base->getNInputs(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getSupervectorLength() const
    { return m_isv_base->getSupervectorLength(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const
    { return m_isv_base->getDimRu(); }

    /**
     * @brief Returns the x session factor
     */
    const blitz::Array<double,1>& getX() const
    { return m_cache_x; }

    /**
     * @brief Returns the z speaker factor
     */
    const blitz::Array<double,1>& getZ() const
    { return m_z; }

    /**
     * @brief Returns the z speaker factors in order to update it
     */
    blitz::Array<double,1>& updateZ()
    { return m_z; }

    /**
     * @brief Returns the V matrix
     */
    void setZ(const blitz::Array<double,1>& z);


    /**
     * @brief Sets the session variable
     */
    void setX(const blitz::Array<double,1>& x);


    /**
     * @brief Returns the ISVBase
     */
    const boost::shared_ptr<bob::learn::em::ISVBase> getISVBase() const
    { return m_isv_base; }

    /**
     * @brief Sets the ISVBase
     */
    void setISVBase(const boost::shared_ptr<bob::learn::em::ISVBase> isv_base);


    /**
     * @brief Estimates x from the GMM statistics considering the LPT
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::learn::em::GMMStats& gmm_stats, blitz::Array<double,1>& x) const
    { m_isv_base->estimateX(gmm_stats, x); }
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
     * @warning Inputs are NOT checked
     * @return score value computed by the machine
     */
    double forward_(const bob::learn::em::GMMStats& input);

  private:
    /**
     * @brief Resize latent variable according to the ISVBase
     */
    void resize();
    /**
     * @ Update cache
     */
    void updateCache();
    /**
     * @brief Resize working arrays
     */
    void resizeTmp();

    // UBM
    boost::shared_ptr<bob::learn::em::ISVBase> m_isv_base;

    // y and z vectors/factors learned during the enrollment procedure
    blitz::Array<double,1> m_z;

    // cache
    blitz::Array<double,1> m_cache_mDz;
    mutable blitz::Array<double,1> m_cache_x;

    // x vector/factor in cache when computing scores
    mutable blitz::Array<double,1> m_tmp_Ux;
};

} } } // namespaces

#endif // BOB_LEARN_EM_ISVMACHINE_H

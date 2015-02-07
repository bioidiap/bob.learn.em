/**
 * @date Tue Jan 27 15:54:00 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * @brief A base class for Joint Factor Analysis-like machines
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_JFABASE_H
#define BOB_LEARN_EM_JFABASE_H

#include <stdexcept>

#include <bob.learn.em/GMMMachine.h>
#include <bob.learn.em/FABase.h>
//#include <bob.learn.em/LinearScoring.h>

#include <bob.io.base/HDF5File.h>
#include <boost/shared_ptr.hpp>

namespace bob { namespace learn { namespace em {


/**
 * @brief A JFA Base class which contains U, V and D matrices
 * TODO: add a reference to the journal articles
 */
class JFABase
{
  public:
    /**
     * @brief Default constructor. Builds a 1 x 1 JFABase
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    JFABase();

    /**
     * @brief Constructor. Builds a new JFABase.
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     *
     * @param ubm The Universal Background Model
     * @param ru size of U (CD x ru)
     * @param rv size of U (CD x rv)
     * @warning ru and rv SHOULD BE  >= 1.
     */
    JFABase(const boost::shared_ptr<bob::learn::em::GMMMachine> ubm, const size_t ru=1, const size_t rv=1);

    /**
     * @brief Copy constructor
     */
    JFABase(const JFABase& other);

    /**
     * @deprecated Starts a new JFAMachine from an existing Configuration object.
     */
    JFABase(bob::io::base::HDF5File& config);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~JFABase();

    /**
     * @brief Assigns from a different JFA machine
     */
    JFABase& operator=(const JFABase &other);

    /**
     * @brief Equal to
     */
    bool operator==(const JFABase& b) const
    { return m_base.operator==(b.m_base); }

    /**
     * @brief Not equal to
     */
    bool operator!=(const JFABase& b) const
    { return m_base.operator!=(b.m_base); }

    /**
     * @brief Similar to
     */
    bool is_similar_to(const JFABase& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const
    { return m_base.is_similar_to(b.m_base, r_epsilon, a_epsilon); }

    /**
     * @brief Saves model to an HDF5 file
     */
    void save(bob::io::base::HDF5File& config) const;

    /**
     * @brief Loads data from an existing configuration object. Resets
     * the current state.
     */
    void load(bob::io::base::HDF5File& config);

    /**
     * @brief Returns the UBM
     */
    const boost::shared_ptr<bob::learn::em::GMMMachine> getUbm() const
    { return m_base.getUbm(); }

    /**
     * @brief Returns the U matrix
     */
    const blitz::Array<double,2>& getU() const
    { return m_base.getU(); }

    /**
     * @brief Returns the V matrix
     */
    const blitz::Array<double,2>& getV() const
    { return m_base.getV(); }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector)
     */
    const blitz::Array<double,1>& getD() const
    { return m_base.getD(); }

    /**
     * @brief Returns the number of Gaussian components
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getNGaussians() const
    { return m_base.getNGaussians();}

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getNInputs() const
    { return m_base.getNInputs(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getSupervectorLength() const
    { return m_base.getSupervectorLength(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const
    { return m_base.getDimRu(); }

    /**
     * @brief Returns the size/rank rv of the V matrix
     */
    const size_t getDimRv() const
    { return m_base.getDimRv(); }

    /**
     * @brief Resets the dimensionality of the subspace U and V
     * U and V are hence uninitialized.
     */
    void resize(const size_t ru, const size_t rv)
    { m_base.resize(ru, rv); }

    /**
     * @brief Returns the U matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason,
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateU()
    { return m_base.updateU(); }

    /**
     * @brief Returns the V matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason,
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateV()
    { return m_base.updateV(); }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector) in order
     * to update it
     * @warning Should only be used by the trainer for efficiency reason,
     *   or for testing purpose.
     */
    blitz::Array<double,1>& updateD()
    { return m_base.updateD(); }


    /**
     * @brief Sets (the mean supervector of) the Universal Background Model
     * U, V and d are uninitialized in case of dimensions update (C or D)
     */
    void setUbm(const boost::shared_ptr<bob::learn::em::GMMMachine> ubm)
    { m_base.setUbm(ubm); }

    /**
     * @brief Sets the U matrix
     */
    void setU(const blitz::Array<double,2>& U)
    { m_base.setU(U); }

    /**
     * @brief Sets the V matrix
     */
    void setV(const blitz::Array<double,2>& V)
    { m_base.setV(V); }

    /**
     * @brief Sets the diagonal matrix diag(d)
     * (a 1D vector is expected as an argument)
     */
    void setD(const blitz::Array<double,1>& d)
    { m_base.setD(d); }

    /**
     * @brief Estimates x from the GMM statistics considering the LPT
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::learn::em::GMMStats& gmm_stats, blitz::Array<double,1>& x) const
    { m_base.estimateX(gmm_stats, x); }

    /**
     * @brief Precompute (put U^{T}.Sigma^{-1} matrix in cache)
     * @warning Should only be used by the trainer for efficiency reason,
     *   or for testing purpose.
     */
    void precompute()
    { m_base.updateCacheUbmUVD(); }

    /**
     * @brief Returns the FABase member
     */
    const bob::learn::em::FABase& getBase() const
    { return m_base; }


  private:
    // FABase
    bob::learn::em::FABase m_base;
};


} } } // namespaces

#endif // BOB_LEARN_EM_JFABASE_H

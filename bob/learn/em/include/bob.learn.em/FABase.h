/**
 * @date Tue Jan 27 15:51:15 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * @brief A base class for Factor Analysis
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_FABASE_H
#define BOB_LEARN_EM_FABASE_H

#include <stdexcept>

#include <bob.learn.em/GMMMachine.h>
#include <boost/shared_ptr.hpp>

namespace bob { namespace learn { namespace em {

/**
 * @brief A FA Base class which contains U, V and D matrices
 * TODO: add a reference to the journal articles
 */
class FABase
{
  public:
    /**
     * @brief Default constructor. Builds an otherwise invalid 0 x 0 FABase
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    FABase();

    /**
     * @brief Constructor. Builds a new FABase.
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     *
     * @param ubm The Universal Background Model
     * @param ru size of U (CD x ru)
     * @param rv size of U (CD x rv)
     * @warning ru and rv SHOULD BE  >= 1. Just set U/V/D to zero if you want
     *   to ignore one subspace. This is the case for ISV.
     */
    FABase(const boost::shared_ptr<bob::learn::em::GMMMachine> ubm, const size_t ru=1, const size_t rv=1);

    /**
     * @brief Copy constructor
     */
    FABase(const FABase& other);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~FABase();

    /**
     * @brief Assigns from a different JFA machine
     */
    FABase& operator=(const FABase &other);

    /**
     * @brief Equal to
     */
    bool operator==(const FABase& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const FABase& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const FABase& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Returns the UBM
     */
    const boost::shared_ptr<bob::learn::em::GMMMachine> getUbm() const
    { return m_ubm; }

    /**
     * @brief Returns the U matrix
     */
    const blitz::Array<double,2>& getU() const
    { return m_U; }

    /**
     * @brief Returns the V matrix
     */
    const blitz::Array<double,2>& getV() const
    { return m_V; }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector)
     */
    const blitz::Array<double,1>& getD() const
    { return m_d; }

    /**
     * @brief Returns the UBM mean supervector (as a 1D vector)
     */
    const blitz::Array<double,1>& getUbmMean() const
    { return m_cache_mean; }

    /**
     * @brief Returns the UBM variance supervector (as a 1D vector)
     */
    const blitz::Array<double,1>& getUbmVariance() const
    { return m_cache_sigma; }

    /**
     * @brief Returns the number of Gaussian components C
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getNGaussians() const
    { if(!m_ubm) throw std::runtime_error("No UBM was set in the JFA machine.");
      return m_ubm->getNGaussians(); }

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getNInputs() const
    { if(!m_ubm) throw std::runtime_error("No UBM was set in the JFA machine.");
      return m_ubm->getNInputs(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has
     *   been set yet.
     */
    const size_t getSupervectorLength() const
    { if(!m_ubm) throw std::runtime_error("No UBM was set in the JFA machine.");
      return m_ubm->getNInputs()*m_ubm->getNGaussians(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const
    { return m_ru; }

    /**
     * @brief Returns the size/rank rv of the V matrix
     */
    const size_t getDimRv() const
    { return m_rv; }

    /**
     * @brief Resets the dimensionality of the subspace U and V
     * U and V are hence uninitialized.
     */
    void resize(const size_t ru, const size_t rv);

    /**
     * @brief Resets the dimensionality of the subspace U and V,
     * assuming that no UBM has yet been set
     * U and V are hence uninitialized.
     */
    void resize(const size_t ru, const size_t rv, const size_t cd);

    /**
     * @brief Returns the U matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason,
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateU()
    { return m_U; }

    /**
     * @brief Returns the V matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason,
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateV()
    { return m_V; }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector) in order
     * to update it
     * @warning Should only be used by the trainer for efficiency reason,
     *   or for testing purpose.
     */
    blitz::Array<double,1>& updateD()
    { return m_d; }


    /**
     * @brief Sets (the mean supervector of) the Universal Background Model
     * U, V and d are uninitialized in case of dimensions update (C or D)
     */
    void setUbm(const boost::shared_ptr<bob::learn::em::GMMMachine> ubm);

    /**
     * @brief Sets the U matrix
     */
    void setU(const blitz::Array<double,2>& U);

    /**
     * @brief Sets the V matrix
     */
    void setV(const blitz::Array<double,2>& V);

    /**
     * @brief Sets the diagonal matrix diag(d)
     * (a 1D vector is expected as an argument)
     */
    void setD(const blitz::Array<double,1>& d);


    /**
     * @brief Estimates x from the GMM statistics considering the LPT
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::learn::em::GMMStats& gmm_stats, blitz::Array<double,1>& x) const;

    /**
     * @brief Compute and put U^{T}.Sigma^{-1} matrix in cache
     * @warning Should only be used by the trainer for efficiency reason,
     *   or for testing purpose.
     */
    void updateCacheUbmUVD();


  private:
    /**
     * @brief Update cache arrays/variables
     */
    void updateCache();
    /**
     * @brief Put GMM mean/variance supervector in cache
     */
    void updateCacheUbm();
    /**
     * @brief Resize working arrays
     */
    void resizeTmp();
    /**
     * @brief Computes (Id + U^T.Sigma^-1.U.N_{i,h}.U)^-1 =
     *   (Id + sum_{c=1..C} N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c})^-1
     */
    void computeIdPlusUSProdInv(const bob::learn::em::GMMStats& gmm_stats,
      blitz::Array<double,2>& out) const;
    /**
     * @brief Computes Fn_x = sum_{sessions h}(N*(o - m))
     * (Normalised first order statistics)
     */
    void computeFn_x(const bob::learn::em::GMMStats& gmm_stats,
      blitz::Array<double,1>& out) const;
    /**
     * @brief Estimates the value of x from the passed arguments
     * (IdPlusUSProdInv and Fn_x), considering the LPT assumption
     */
    void estimateX(const blitz::Array<double,2>& IdPlusUSProdInv,
      const blitz::Array<double,1>& Fn_x, blitz::Array<double,1>& x) const;


    // UBM
    boost::shared_ptr<bob::learn::em::GMMMachine> m_ubm;

    // dimensionality
    size_t m_ru; // size of U (CD x ru)
    size_t m_rv; // size of V (CD x rv)

    // U, V, D matrices
    // D is assumed to be diagonal, and only the diagonal is stored
    blitz::Array<double,2> m_U;
    blitz::Array<double,2> m_V;
    blitz::Array<double,1> m_d;

    // Vectors/Matrices precomputed in cache
    blitz::Array<double,1> m_cache_mean;
    blitz::Array<double,1> m_cache_sigma;
    blitz::Array<double,2> m_cache_UtSigmaInv;

    mutable blitz::Array<double,2> m_tmp_IdPlusUSProdInv;
    mutable blitz::Array<double,1> m_tmp_Fn_x;
    mutable blitz::Array<double,1> m_tmp_ru;
    mutable blitz::Array<double,2> m_tmp_ruD;
    mutable blitz::Array<double,2> m_tmp_ruru;
};


} } } // namespaces

#endif // BOB_LEARN_EM_FABASE_H

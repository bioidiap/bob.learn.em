/**
 * @date Sat Jan 31 17:16:17 2015 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * @brief FABaseTrainer functions
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_EM_FABASETRAINER_H
#define BOB_LEARN_EM_FABASETRAINER_H

#include <blitz/array.h>
#include <bob.learn.em/GMMStats.h>
#include <bob.learn.em/JFAMachine.h>
#include <vector>

#include <map>
#include <string>
#include <bob.core/array_copy.h>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <bob.core/logging.h>

namespace bob { namespace learn { namespace em {

class FABaseTrainer
{
  public:
    /**
     * @brief Constructor
     */
    FABaseTrainer();

    /**
     * @brief Copy constructor
     */
    FABaseTrainer(const FABaseTrainer& other);

    /**
     * @brief Destructor
     */
    ~FABaseTrainer();

    /**
     * @brief Check that the dimensionality of the statistics match.
     */
    void checkStatistics(const bob::learn::em::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);

    /**
     * @brief Initialize the dimensionality, the UBM, the sums of the
     * statistics and the number of identities.
     */
    void initUbmNidSumStatistics(const bob::learn::em::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);

    /**
     * @brief Precomputes the sums of the zeroth order statistics over the
     * sessions for each client
     */
    void precomputeSumStatisticsN(const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);
    /**
     * @brief Precomputes the sums of the first order statistics over the
     * sessions for each client
     */
    void precomputeSumStatisticsF(const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);

    /**
     * @brief Initializes (allocates and sets to zero) the x, y, z speaker
     * factors
     */
    void initializeXYZ(const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);

    /**
     * @brief Resets the x, y, z speaker factors to zero values
     */
    void resetXYZ();


    /**** Y and V functions ****/
    /**
     * @brief Computes Vt * diag(sigma)^-1
     */
    void computeVtSigmaInv(const bob::learn::em::FABase& m);
    /**
     * @brief Computes Vt_{c} * diag(sigma)^-1 * V_{c} for each Gaussian c
     */
    void computeVProd(const bob::learn::em::FABase& m);
    /**
     * @brief Computes (I+Vt*diag(sigma)^-1*Ni*V)^-1 which occurs in the y
     * estimation for the given person
     */
    void computeIdPlusVProd_i(const size_t id);
    /**
     * @brief Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h})
     * which occurs in the y estimation of the given person
     */
    void computeFn_y_i(const bob::learn::em::FABase& m,
      const std::vector<boost::shared_ptr<bob::learn::em::GMMStats> >& stats,
      const size_t id);
    /**
     * @brief Updates y_i (of the current person) and the accumulators to
     * compute V with the cache values m_cache_IdPlusVprod_i, m_VtSigmaInv and
     * m_cache_Fn_y_i
     */
    void updateY_i(const size_t id);
    /**
     * @brief Updates y and the accumulators to compute V
     */
    void updateY(const bob::learn::em::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);
    /**
     * @brief Computes the accumulators m_acc_V_A1 and m_acc_V_A2 for V
     * V = A2 * A1^-1
     */
    void computeAccumulatorsV(const bob::learn::em::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);
    /**
     * @brief Updates V from the accumulators m_acc_V_A1 and m_acc_V_A2
     */
    void updateV(blitz::Array<double,2>& V);


    /**** X and U functions ****/
    /**
     * @brief Computes Ut * diag(sigma)^-1
     */
    void computeUtSigmaInv(const bob::learn::em::FABase& m);
    /**
     * @brief Computes Ut_{c} * diag(sigma)^-1 * U_{c} for each Gaussian c
     */
    void computeUProd(const bob::learn::em::FABase& m);
    /**
     * @brief Computes (I+Ut*diag(sigma)^-1*Ni*U)^-1 which occurs in the x
     * estimation
     */
    void computeIdPlusUProd_ih(const boost::shared_ptr<bob::learn::em::GMMStats>& stats);
    /**
     * @brief Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h})
     * which occurs in the y estimation of the given person
     */
    void computeFn_x_ih(const bob::learn::em::FABase& m,
      const boost::shared_ptr<bob::learn::em::GMMStats>& stats, const size_t id);
    /**
     * @brief Updates x_ih (of the current person/session) and the
     * accumulators to compute U with the cache values m_cache_IdPlusVprod_i,
     * m_VtSigmaInv and m_cache_Fn_y_i
     */
    void updateX_ih(const size_t id, const size_t h);
    /**
     * @brief Updates x
     */
    void updateX(const bob::learn::em::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);
    /**
     * @brief Computes the accumulators m_acc_U_A1 and m_acc_U_A2 for U
     * U = A2 * A1^-1
     */
    void computeAccumulatorsU(const bob::learn::em::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);
    /**
     * @brief Updates U from the accumulators m_acc_U_A1 and m_acc_U_A2
     */
    void updateU(blitz::Array<double,2>& U);


    /**** z and D functions ****/
    /**
     * @brief Computes diag(D) * diag(sigma)^-1
     */
    void computeDtSigmaInv(const bob::learn::em::FABase& m);
    /**
     * @brief Computes Dt_{c} * diag(sigma)^-1 * D_{c} for each Gaussian c
     */
    void computeDProd(const bob::learn::em::FABase& m);
    /**
     * @brief Computes (I+diag(d)t*diag(sigma)^-1*Ni*diag(d))^-1 which occurs
     * in the z estimation for the given person
     */
    void computeIdPlusDProd_i(const size_t id);
    /**
     * @brief Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h})
     * which occurs in the y estimation of the given person
     */
    void computeFn_z_i(const bob::learn::em::FABase& m,
      const std::vector<boost::shared_ptr<bob::learn::em::GMMStats> >& stats, const size_t id);
    /**
     * @brief Updates z_i (of the current person) and the accumulators to
     * compute D with the cache values m_cache_IdPlusDProd_i, m_VtSigmaInv
     * and m_cache_Fn_z_i
     */
    void updateZ_i(const size_t id);
    /**
     * @brief Updates z and the accumulators to compute D
     */
    void updateZ(const bob::learn::em::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);
    /**
     * @brief Computes the accumulators m_acc_D_A1 and m_acc_D_A2 for d
     * d = A2 * A1^-1
     */
    void computeAccumulatorsD(const bob::learn::em::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& stats);
    /**
     * @brief Updates d from the accumulators m_acc_D_A1 and m_acc_D_A2
     */
    void updateD(blitz::Array<double,1>& d);


    /**
     * @brief Get the zeroth order statistics
     */
    const std::vector<blitz::Array<double,1> >& getNacc() const
    { return m_Nacc; }
    /**
     * @brief Get the first order statistics
     */
    const std::vector<blitz::Array<double,1> >& getFacc() const
    { return m_Facc; }
    /**
     * @brief Get the x speaker factors
     */
    const std::vector<blitz::Array<double,2> >& getX() const
    { return m_x; }
    /**
     * @brief Get the y speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getY() const
    { return m_y; }
    /**
     * @brief Get the z speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getZ() const
    { return m_z; }
    /**
     * @brief Set the x speaker factors
     */
    void setX(const std::vector<blitz::Array<double,2> >& X)
    { m_x = X; }
    /**
     * @brief Set the y speaker factors
     */
    void setY(const std::vector<blitz::Array<double,1> >& y)
    { m_y = y; }
    /**
     * @brief Set the z speaker factors
     */
    void setZ(const std::vector<blitz::Array<double,1> >& z)
    { m_z = z; }

    /**
     * @brief Initializes the cache to process the given statistics
     */
    void initCache();

    /**
     * @brief Getters for the accumulators
     */
    const blitz::Array<double,3>& getAccVA1() const
    { return m_acc_V_A1; }
    const blitz::Array<double,2>& getAccVA2() const
    { return m_acc_V_A2; }
    const blitz::Array<double,3>& getAccUA1() const
    { return m_acc_U_A1; }
    const blitz::Array<double,2>& getAccUA2() const
    { return m_acc_U_A2; }
    const blitz::Array<double,1>& getAccDA1() const
    { return m_acc_D_A1; }
    const blitz::Array<double,1>& getAccDA2() const
    { return m_acc_D_A2; }

    /**
     * @brief Setters for the accumulators, Very useful if the e-Step needs
     * to be parallelized.
     */
    void setAccVA1(const blitz::Array<double,3>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_V_A1);
      m_acc_V_A1 = acc; }
    void setAccVA2(const blitz::Array<double,2>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_V_A2);
      m_acc_V_A2 = acc; }
    void setAccUA1(const blitz::Array<double,3>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_U_A1);
      m_acc_U_A1 = acc; }
    void setAccUA2(const blitz::Array<double,2>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_U_A2);
      m_acc_U_A2 = acc; }
    void setAccDA1(const blitz::Array<double,1>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_D_A1);
      m_acc_D_A1 = acc; }
    void setAccDA2(const blitz::Array<double,1>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_D_A2);
      m_acc_D_A2 = acc; }


  private:
    size_t m_Nid; // Number of identities
    size_t m_dim_C; // Number of Gaussian components of the UBM GMM
    size_t m_dim_D; // Dimensionality of the feature space
    size_t m_dim_ru; // Rank of the U subspace
    size_t m_dim_rv; // Rank of the V subspace

    std::vector<blitz::Array<double,2> > m_x; // matrix x of speaker factors for eigenchannels U, for each client
    std::vector<blitz::Array<double,1> > m_y; // vector y of spealer factors for eigenvoices V, for each client
    std::vector<blitz::Array<double,1> > m_z; // vector z of spealer factors for eigenvoices Z, for each client

    std::vector<blitz::Array<double,1> > m_Nacc; // Sum of the zeroth order statistics over the sessions for each client, dimension C
    std::vector<blitz::Array<double,1> > m_Facc; // Sum of the first order statistics over the sessions for each client, dimension CD

    // Accumulators for the M-step
    blitz::Array<double,3> m_acc_V_A1;
    blitz::Array<double,2> m_acc_V_A2;
    blitz::Array<double,3> m_acc_U_A1;
    blitz::Array<double,2> m_acc_U_A2;
    blitz::Array<double,1> m_acc_D_A1;
    blitz::Array<double,1> m_acc_D_A2;

    // Cache/Precomputation
    blitz::Array<double,2> m_cache_VtSigmaInv; // Vt * diag(sigma)^-1
    blitz::Array<double,3> m_cache_VProd; // first dimension is the Gaussian id
    blitz::Array<double,2> m_cache_IdPlusVProd_i;
    blitz::Array<double,1> m_cache_Fn_y_i;

    blitz::Array<double,2> m_cache_UtSigmaInv; // Ut * diag(sigma)^-1
    blitz::Array<double,3> m_cache_UProd; // first dimension is the Gaussian id
    blitz::Array<double,2> m_cache_IdPlusUProd_ih;
    blitz::Array<double,1> m_cache_Fn_x_ih;

    blitz::Array<double,1> m_cache_DtSigmaInv; // Dt * diag(sigma)^-1
    blitz::Array<double,1> m_cache_DProd; // supervector length dimension
    blitz::Array<double,1> m_cache_IdPlusDProd_i;
    blitz::Array<double,1> m_cache_Fn_z_i;

    // Working arrays
    mutable blitz::Array<double,2> m_tmp_ruru;
    mutable blitz::Array<double,2> m_tmp_ruD;
    mutable blitz::Array<double,2> m_tmp_rvrv;
    mutable blitz::Array<double,2> m_tmp_rvD;
    mutable blitz::Array<double,1> m_tmp_rv;
    mutable blitz::Array<double,1> m_tmp_ru;
    mutable blitz::Array<double,1> m_tmp_CD;
    mutable blitz::Array<double,1> m_tmp_CD_b;
};


} } } // namespaces

#endif /* BOB_LEARN_EM_FABASETRAINER_H */

/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Fri Oct 14 18:07:56 2011 +0200
 *
 * @brief Python bindings for the PLDABase/PLDAMachine
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>

#include "ndarray.h"
#include <bob.learn.misc/PLDAMachine.h>

using namespace boost::python;

static void py_set_dim_d(bob::learn::misc::PLDABase& machine, const size_t dim_d)
{
  machine.resize(dim_d, machine.getDimF(), machine.getDimG());
}
static void py_set_dim_f(bob::learn::misc::PLDABase& machine, const size_t dim_f)
{
  machine.resize(machine.getDimD(), dim_f, machine.getDimG());
}
static void py_set_dim_g(bob::learn::misc::PLDABase& machine, const size_t dim_g)
{
  machine.resize(machine.getDimD(), machine.getDimF(), dim_g);
}

// Set methods that uses blitz::Arrays
static void py_set_mu(bob::learn::misc::PLDABase& machine,
  bob::python::const_ndarray mu)
{
  machine.setMu(mu.bz<double,1>());
}

static void py_set_f(bob::learn::misc::PLDABase& machine,
  bob::python::const_ndarray f)
{
  machine.setF(f.bz<double,2>());
}

static void py_set_g(bob::learn::misc::PLDABase& machine,
  bob::python::const_ndarray g)
{
  machine.setG(g.bz<double,2>());
}

static void py_set_sigma(bob::learn::misc::PLDABase& machine,
  bob::python::const_ndarray sigma)
{
  machine.setSigma(sigma.bz<double,1>());
}


static double computeLogLikelihood(bob::learn::misc::PLDAMachine& plda,
  bob::python::const_ndarray samples, bool with_enrolled_samples=true)
{
  const bob::io::base::array::typeinfo& info = samples.type();
  switch (info.nd) {
    case 1:
      return plda.computeLogLikelihood(samples.bz<double,1>(), with_enrolled_samples);
    case 2:
      return plda.computeLogLikelihood(samples.bz<double,2>(), with_enrolled_samples);
    default:
      PYTHON_ERROR(TypeError, "PLDA log-likelihood computation does not accept input array with '" SIZE_T_FMT "' dimensions (only 1D or 2D arrays)", info.nd);
  }
}

static double plda_forward_sample(bob::learn::misc::PLDAMachine& m,
  bob::python::const_ndarray samples)
{
  const bob::io::base::array::typeinfo& info = samples.type();
  switch (info.nd) {
    case 1:
      {
        double score;
        // Calls the forward function
        m.forward(samples.bz<double,1>(), score);
        return score;
      }
    case 2:
      {
        double score;
        // Calls the forward function
        m.forward(samples.bz<double,2>(), score);
        return score;
      }
    default:
      PYTHON_ERROR(TypeError, "PLDA forwarding does not accept input array with '" SIZE_T_FMT "' dimensions (only 1D or 2D arrays)", info.nd);
  }
}

static double py_log_likelihood_point_estimate(bob::learn::misc::PLDABase& plda,
  bob::python::const_ndarray xij, bob::python::const_ndarray hi,
  bob::python::const_ndarray wij)
{
  return plda.computeLogLikelihoodPointEstimate(xij.bz<double,1>(),
           hi.bz<double,1>(), wij.bz<double,1>());
}

BOOST_PYTHON_FUNCTION_OVERLOADS(computeLogLikelihood_overloads, computeLogLikelihood, 2, 3)


static boost::shared_ptr<bob::learn::misc::PLDABase> b_init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::PLDABase>(new bob::learn::misc::PLDABase(*hdf5->f));
}

static void b_load(bob::learn::misc::PLDABase& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void b_save(const bob::learn::misc::PLDABase& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}


static boost::shared_ptr<bob::learn::misc::PLDAMachine> m_init(boost::python::object file, boost::shared_ptr<bob::learn::misc::PLDABase> b){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::PLDAMachine>(new bob::learn::misc::PLDAMachine(*hdf5->f, b));
}

static void m_load(bob::learn::misc::PLDAMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void m_save(const bob::learn::misc::PLDAMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}

void bind_machine_plda()
{
  class_<bob::learn::misc::PLDABase, boost::shared_ptr<bob::learn::misc::PLDABase> >("PLDABase", "A PLDABase can be seen as a container for the subspaces F, G, the diagonal covariance matrix sigma (stored as a 1D array) and the mean vector mu when performing Probabilistic Linear Discriminant Analysis (PLDA). PLDA is a probabilistic model that incorporates components describing both between-class and within-class variations. A PLDABase can be shared between several PLDAMachine that contains class-specific information (information about the enrolment samples).\n\nReferences:\n1. 'A Scalable Formulation of Probabilistic Linear Discriminant Analysis: Applied to Face Recognition', Laurent El Shafey, Chris McCool, Roy Wallace, Sebastien Marcel, TPAMI'2013\n2. 'Probabilistic Linear Discriminant Analysis for Inference About Identity', Prince and Elder, ICCV'2007.\n3. 'Probabilistic Models for Inference about Identity', Li, Fu, Mohammed, Elder and Prince, TPAMI'2012.", init<const size_t, const size_t, const size_t, optional<const double> >((arg("self"), arg("dim_d"), arg("dim_f"), arg("dim_g"), arg("variance_flooring")=0.), "Builds a new PLDABase. dim_d is the dimensionality of the input features, dim_f is the dimensionality of the F subspace and dim_g the dimensionality of the G subspace. The variance flooring threshold is the minimum value that the variance sigma can reach, as this diagonal matrix is inverted."))
    .def(init<>((arg("self")), "Constructs a new empty PLDABase."))
    .def("__init__", boost::python::make_constructor(&b_init), "Constructs a new PLDABase from a configuration file.")
    .def(init<const bob::learn::misc::PLDABase&>((arg("self"), arg("machine")), "Copy constructs a PLDABase"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::PLDABase::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this PLDABase with the 'other' one to be approximately the same.")
    .def("load", &b_load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &b_save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .add_property("dim_d", &bob::learn::misc::PLDABase::getDimD, &py_set_dim_d, "Dimensionality of the input feature vectors")
    .add_property("dim_f", &bob::learn::misc::PLDABase::getDimF, &py_set_dim_f, "Dimensionality of the F subspace/matrix of the PLDA model")
    .add_property("dim_g", &bob::learn::misc::PLDABase::getDimG, &py_set_dim_g, "Dimensionality of the G subspace/matrix of the PLDA model")
    .add_property("mu", make_function(&bob::learn::misc::PLDABase::getMu, return_value_policy<copy_const_reference>()), &py_set_mu, "The mean vector mu of the PLDA model")
    .add_property("f", make_function(&bob::learn::misc::PLDABase::getF, return_value_policy<copy_const_reference>()), &py_set_f, "The subspace/matrix F of the PLDA model")
    .add_property("g", make_function(&bob::learn::misc::PLDABase::getG, return_value_policy<copy_const_reference>()), &py_set_g, "The subspace/matrix G of the PLDA model")
    .add_property("sigma", make_function(&bob::learn::misc::PLDABase::getSigma, return_value_policy<copy_const_reference>()), &py_set_sigma, "The diagonal covariance matrix (represented by a 1D numpy array) sigma of the PLDA model")
    .add_property("variance_threshold", &bob::learn::misc::PLDABase::getVarianceThreshold, &bob::learn::misc::PLDABase::setVarianceThreshold,
      "The variance flooring threshold, i.e. the minimum allowed value of variance (sigma) in each dimension. "
      "The variance sigma will be set to this value if an attempt is made to set it to a smaller value.")
    .def("resize", &bob::learn::misc::PLDABase::resize, (arg("self"), arg("dim_d"), arg("dim_f"), arg("dim_g")), "Resizes the dimensionality of the PLDA model. Paramaters mu, F, G and sigma are reinitialized.")
    .def("has_gamma", &bob::learn::misc::PLDABase::hasGamma, (arg("self"), arg("a")), "Tells if the gamma matrix for the given number of samples has already been computed. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("compute_gamma", &bob::learn::misc::PLDABase::computeGamma, (arg("self"), arg("a"), arg("gamma")), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_add_gamma", make_function(&bob::learn::misc::PLDABase::getAddGamma, return_value_policy<copy_const_reference>(), (arg("self"), arg("a"))), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_gamma", make_function(&bob::learn::misc::PLDABase::getGamma, return_value_policy<copy_const_reference>(), (arg("self"), arg("a"))), "Returns the gamma matrix for the given number of samples if it has already been put in cache. Throws an exception otherwise. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("has_log_like_const_term", &bob::learn::misc::PLDABase::hasLogLikeConstTerm, (arg("self"), arg("a")), "Tells if the log likelihood constant term for the given number of samples has already been computed.")
    .def("compute_log_like_const_term", (double (bob::learn::misc::PLDABase::*)(const size_t, const blitz::Array<double,2>&) const)&bob::learn::misc::PLDABase::computeLogLikeConstTerm, (arg("self"), arg("a"), arg("gamma")), "Computes the log likelihood constant term for the given number of samples.")
    .def("get_add_log_like_const_term", &bob::learn::misc::PLDABase::getAddLogLikeConstTerm, (arg("self"), arg("a")), "Computes the log likelihood constant term for the given number of samples, and adds it to the machine (as well as gamma), if it does not already exist.")
    .def("get_log_like_const_term", &bob::learn::misc::PLDABase::getLogLikeConstTerm, (arg("self"), arg("a")), "Returns the log likelihood constant term for the given number of samples if it has already been put in cache. Throws an exception otherwise.")
    .def("clear_maps", &bob::learn::misc::PLDABase::clearMaps, (arg("self")), "Clear the maps containing the gamma's as well as the log likelihood constant term for few number of samples. These maps are used to make likelihood computations faster.")
    .def("compute_log_likelihood_point_estimate", &py_log_likelihood_point_estimate, (arg("self"), arg("xij"), arg("hi"), arg("wij")), "Computes the log-likelihood of a sample given the latent variables hi and wij (point estimate rather than Bayesian-like full integration).")
    .def(self_ns::str(self_ns::self))
    .add_property("__isigma__", make_function(&bob::learn::misc::PLDABase::getISigma, return_value_policy<copy_const_reference>()), "sigma^{-1} matrix stored in cache")
    .add_property("__alpha__", make_function(&bob::learn::misc::PLDABase::getAlpha, return_value_policy<copy_const_reference>()), "alpha matrix stored in cache")
    .add_property("__beta__", make_function(&bob::learn::misc::PLDABase::getBeta, return_value_policy<copy_const_reference>()), "beta matrix stored in cache")
    .add_property("__ft_beta__", make_function(&bob::learn::misc::PLDABase::getFtBeta, return_value_policy<copy_const_reference>()), "F^T.beta matrix stored in cache")
    .add_property("__gt_i_sigma__", make_function(&bob::learn::misc::PLDABase::getGtISigma, return_value_policy<copy_const_reference>()), "G^T.sigma^{-1} matrix stored in cache")
    .add_property("__logdet_alpha__", &bob::learn::misc::PLDABase::getLogDetAlpha, "Logarithm of the determinant of the alpha matrix stored in cache.")
    .add_property("__logdet_sigma__", &bob::learn::misc::PLDABase::getLogDetSigma, "Logarithm of the determinant of the sigma matrix stored in cache.")
    .def("__precompute__", &bob::learn::misc::PLDABase::precompute, (arg("self")), "Precomputes useful values such as alpha and beta.")
    .def("__precompute_log_like__", &bob::learn::misc::PLDABase::precomputeLogLike, (arg("self")), "Precomputes useful values for log-likelihood computations.")
  ;

  class_<bob::learn::misc::PLDAMachine, boost::shared_ptr<bob::learn::misc::PLDAMachine> >("PLDAMachine", "A PLDAMachine contains class-specific information (from the enrolment samples) when performing Probabilistic Linear Discriminant Analysis (PLDA). It should be attached to a PLDABase that contains information such as the subspaces F and G.\n\nReferences:\n1. 'A Scalable Formulation of Probabilistic Linear Discriminant Analysis: Applied to Face Recognition', Laurent El Shafey, Chris McCool, Roy Wallace, Sebastien Marcel, TPAMI'2013\n2. 'Probabilistic Linear Discriminant Analysis for Inference About Identity', Prince and Elder, ICCV'2007.\n3. 'Probabilistic Models for Inference about Identity', Li, Fu, Mohammed, Elder and Prince, TPAMI'2012.", init<boost::shared_ptr<bob::learn::misc::PLDABase> >((arg("self"), arg("plda_base")), "Builds a new PLDAMachine. An attached PLDABase should be provided, that can be shared by several PLDAMachine."))
    .def(init<>((arg("self")), "Constructs a new empty (invalid) PLDAMachine. A PLDABase should then be set using the 'plda_base' attribute of this object."))
    .def("__init__", make_constructor(&m_init), "Constructs a new PLDAMachine from a configuration file (and a PLDABase object).")
    .def(init<const bob::learn::misc::PLDAMachine&>((arg("self"), arg("machine")), "Copy constructs a PLDAMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::PLDAMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this PLDAMachine with the 'other' one to be approximately the same.")
    .def("load", &m_load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file. The PLDABase will not be loaded, and has to be set manually using the 'plda_base' attribute.")
    .def("save", &m_save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file. The PLDABase will not be saved, and has to be saved separately, as it can be shared by several PLDAMachines.")
    .add_property("plda_base", &bob::learn::misc::PLDAMachine::getPLDABase, &bob::learn::misc::PLDAMachine::setPLDABase)
    .add_property("dim_d", &bob::learn::misc::PLDAMachine::getDimD, "Dimensionality of the input feature vectors")
    .add_property("dim_f", &bob::learn::misc::PLDAMachine::getDimF, "Dimensionality of the F subspace/matrix of the PLDA model")
    .add_property("dim_g", &bob::learn::misc::PLDAMachine::getDimG, "Dimensionality of the G subspace/matrix of the PLDA model")
    .add_property("n_samples", &bob::learn::misc::PLDAMachine::getNSamples, &bob::learn::misc::PLDAMachine::setNSamples, "Number of enrolled samples")
    .add_property("w_sum_xit_beta_xi", &bob::learn::misc::PLDAMachine::getWSumXitBetaXi, &bob::learn::misc::PLDAMachine::setWSumXitBetaXi)
    .add_property("weighted_sum", make_function(&bob::learn::misc::PLDAMachine::getWeightedSum, return_value_policy<copy_const_reference>()), &bob::learn::misc::PLDAMachine::setWeightedSum)
    .add_property("log_likelihood", &bob::learn::misc::PLDAMachine::getLogLikelihood, &bob::learn::misc::PLDAMachine::setLogLikelihood)
    .def("has_gamma", &bob::learn::misc::PLDAMachine::hasGamma, (arg("self"), arg("a")), "Tells if the gamma matrix for the given number of samples has already been computed. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_add_gamma", make_function(&bob::learn::misc::PLDAMachine::getAddGamma, return_value_policy<copy_const_reference>(), (arg("self"), arg("a"))), "Computes the gamma matrix for the given number of samples. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("get_gamma", make_function(&bob::learn::misc::PLDAMachine::getGamma, return_value_policy<copy_const_reference>(), (arg("self"), arg("a"))), "Returns the gamma matrix for the given number of samples if it has already been put in cache. Throws an exception otherwise. (gamma = inverse(I+a.F^T.beta.F), please check the documentation/source code for more details.")
    .def("has_log_like_const_term", &bob::learn::misc::PLDAMachine::hasLogLikeConstTerm, (arg("self"), arg("a")), "Tells if the log likelihood constant term for the given number of samples has already been computed.")
    .def("get_add_log_like_const_term", &bob::learn::misc::PLDAMachine::getAddLogLikeConstTerm, (arg("self"), arg("a")), "Computes the log likelihood constant term for the given number of samples, and adds it to the machine (as well as gamma), if it does not already exist.")
    .def("get_log_like_const_term", &bob::learn::misc::PLDAMachine::getLogLikeConstTerm, (arg("self"), arg("a")), "Returns the log likelihood constant term for the given number of samples if it has already been put in cache. Throws an exception otherwise.")
    .def("clear_maps", &bob::learn::misc::PLDAMachine::clearMaps, (arg("self")), "Clears the maps containing the gamma's as well as the log likelihood constant term for few number of samples. These maps are used to make likelihood computations faster.")
    .def("compute_log_likelihood", &computeLogLikelihood, computeLogLikelihood_overloads((arg("self"), arg("sample"), arg("use_enrolled_samples")=true), "Computes the log-likelihood considering only the probe sample(s) or jointly the probe sample(s) and the enrolled samples."))
    .def("__call__", &plda_forward_sample, (arg("self"), arg("sample")), "Processes a sample and returns a log-likelihood ratio score.")
    .def("forward", &plda_forward_sample, (arg("self"), arg("sample")), "Processes a sample and returns a log-likelihood ratio score.")
  ;
}

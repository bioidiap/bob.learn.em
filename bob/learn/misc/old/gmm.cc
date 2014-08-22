/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue Jul 26 15:11:33 2011 +0200
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>

#include "ndarray.h"

#include <bob.learn.misc/GMMStats.h>
#include <bob.learn.misc/GMMMachine.h>

using namespace boost::python;

static object py_gmmstats_getN(bob::learn::misc::GMMStats& s)
{
  bob::python::ndarray n(bob::io::base::array::t_float64, s.n.extent(0));
  blitz::Array<double,1> n_ = n.bz<double,1>();
  n_ = s.n;
  return n.self();
}

static void py_gmmstats_setN(bob::learn::misc::GMMStats& s,
  bob::python::const_ndarray n)
{
  s.n = n.bz<double,1>();
}

static object py_gmmstats_getSumpx(bob::learn::misc::GMMStats& s)
{
  bob::python::ndarray sumpx(bob::io::base::array::t_float64, s.sumPx.extent(0),
    s.sumPx.extent(1));
  blitz::Array<double,2> sumpx_ = sumpx.bz<double,2>();
  sumpx_ = s.sumPx;
  return sumpx.self();
}

static void py_gmmstats_setSumpx(bob::learn::misc::GMMStats& s,
  bob::python::const_ndarray sumpx)
{
  s.sumPx = sumpx.bz<double,2>();
}

static object py_gmmstats_getSumpxx(bob::learn::misc::GMMStats& s)
{
  bob::python::ndarray sumpxx(bob::io::base::array::t_float64, s.sumPxx.extent(0),
    s.sumPxx.extent(1));
  blitz::Array<double,2> sumpxx_ = sumpxx.bz<double,2>();
  sumpxx_ = s.sumPxx;
  return sumpxx.self();
}

static void py_gmmstats_setSumpxx(bob::learn::misc::GMMStats& s,
  bob::python::const_ndarray sumpxx)
{
  s.sumPxx = sumpxx.bz<double,2>();
}


static void py_gmmmachine_setWeights(bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray weights)
{
  machine.setWeights(weights.bz<double,1>());
}

static object py_gmmmachine_getMeans(const bob::learn::misc::GMMMachine& machine)
{
  bob::python::ndarray means(bob::io::base::array::t_float64,
    machine.getNGaussians(), machine.getNInputs());
  blitz::Array<double,2> means_ = means.bz<double,2>();
  machine.getMeans(means_);
  return means.self();
}

static void py_gmmmachine_setMeans(bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray means)
{
  machine.setMeans(means.bz<double,2>());
}

static void py_gmmmachine_setMeanSupervector(bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray vec)
{
  machine.setMeanSupervector(vec.bz<double,1>());
}

static object py_gmmmachine_getVariances(const bob::learn::misc::GMMMachine& machine)
{
  bob::python::ndarray variances(bob::io::base::array::t_float64,
    machine.getNGaussians(), machine.getNInputs());
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  machine.getVariances(variances_);
  return variances.self();
}

static void py_gmmmachine_setVariances(bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray variances)
{
  machine.setVariances(variances.bz<double,2>());
}

static void py_gmmmachine_setVarianceSupervector(bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray vec)
{
  machine.setVarianceSupervector(vec.bz<double,1>());
}

static object py_gmmmachine_getVarianceThresholds(const bob::learn::misc::GMMMachine& machine)
{
  bob::python::ndarray varianceThresholds(bob::io::base::array::t_float64,
    machine.getNGaussians(), machine.getNInputs());
  blitz::Array<double,2> varianceThresholds_ = varianceThresholds.bz<double,2>();
  machine.getVarianceThresholds(varianceThresholds_);
  return varianceThresholds.self();
}

static void py_gmmmachine_setVarianceThresholds(bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray varianceThresholds)
{
  machine.setVarianceThresholds(varianceThresholds.bz<double,2>());
}

static void py_gmmmachine_setVarianceThresholdsOther(bob::learn::misc::GMMMachine& machine,
  object o)
{
  extract<int> int_check(o);
  extract<double> float_check(o);
  if(int_check.check()) { //is int
    machine.setVarianceThresholds(int_check());
  }
  else if(float_check.check()) { //is float
    machine.setVarianceThresholds(float_check());
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    extract<bob::python::const_ndarray> array_check(o);
    if (!array_check.check())
      PYTHON_ERROR(TypeError, "Cannot extract an array from this Python object");
    bob::python::const_ndarray ar = array_check();
    machine.setVarianceThresholds(ar.bz<double,1>());
  }
}

static tuple py_gmmmachine_get_shape(const bob::learn::misc::GMMMachine& m)
{
  return make_tuple(m.getNGaussians(), m.getNInputs());
}

static void py_gmmmachine_set_shape(bob::learn::misc::GMMMachine& m,
  const blitz::TinyVector<int,2>& s)
{
  m.resize(s(0), s(1));
}

static double py_gmmmachine_loglikelihoodA(const bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray x, bob::python::ndarray ll)
{
  blitz::Array<double,1> ll_ = ll.bz<double,1>();
  return machine.logLikelihood(x.bz<double,1>(), ll_);
}

static double py_gmmmachine_loglikelihoodA_(const bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray x, bob::python::ndarray ll)
{
  blitz::Array<double,1> ll_ = ll.bz<double,1>();
  return machine.logLikelihood_(x.bz<double,1>(), ll_);
}

static double py_gmmmachine_loglikelihoodB(const bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray x)
{
  return machine.logLikelihood(x.bz<double,1>());
}

static double py_gmmmachine_loglikelihoodB_(const bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray x)
{
  return machine.logLikelihood_(x.bz<double,1>());
}

static void py_gmmmachine_accStatistics(const bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray x, bob::learn::misc::GMMStats& gs)
{
  const bob::io::base::array::typeinfo& info = x.type();
  switch(info.nd) {
    case 1:
      machine.accStatistics(x.bz<double,1>(), gs);
      break;
    case 2:
      machine.accStatistics(x.bz<double,2>(), gs);
      break;
    default:
      PYTHON_ERROR(TypeError, "cannot accStatistics of arrays with "  SIZE_T_FMT " dimensions (only with 1 or 2 dimensions).", info.nd);
  }
}

static void py_gmmmachine_accStatistics_(const bob::learn::misc::GMMMachine& machine,
  bob::python::const_ndarray x, bob::learn::misc::GMMStats& gs)
{
  const bob::io::base::array::typeinfo& info = x.type();
  switch(info.nd) {
    case 1:
      machine.accStatistics_(x.bz<double,1>(), gs);
      break;
    case 2:
      machine.accStatistics_(x.bz<double,2>(), gs);
      break;
    default:
      PYTHON_ERROR(TypeError, "cannot accStatistics of arrays with "  SIZE_T_FMT " dimensions (only with 1 or 2 dimensions).", info.nd);
  }
}

static boost::shared_ptr<bob::learn::misc::GMMStats> s_init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::GMMStats>(new bob::learn::misc::GMMStats(*hdf5->f));
}

static void s_load(bob::learn::misc::GMMStats& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void s_save(const bob::learn::misc::GMMStats& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}


static boost::shared_ptr<bob::learn::misc::GMMMachine> m_init(boost::python::object file){
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::GMMMachine>(new bob::learn::misc::GMMMachine(*hdf5->f));
}

static void m_load(bob::learn::misc::GMMMachine& self, boost::python::object file){
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void m_save(const bob::learn::misc::GMMMachine& self, boost::python::object file){
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}

void bind_machine_gmm()
{
  class_<bob::learn::misc::GMMStats, boost::shared_ptr<bob::learn::misc::GMMStats> >("GMMStats",
      "A container for GMM statistics.\n"
      "With respect to Reynolds, \"Speaker Verification Using Adapted "
      "Gaussian Mixture Models\", DSP, 2000:\n"
      "Eq (8) is n(i)\n"
      "Eq (9) is sumPx(i) / n(i)\n"
      "Eq (10) is sumPxx(i) / n(i)\n",
      init<>(arg("self")))
    .def("__init__", boost::python::make_constructor(&s_init))
    .def(init<const size_t, const size_t>((arg("self"), arg("n_gaussians"), arg("n_inputs"))))
    .def(init<bob::learn::misc::GMMStats&>((arg("self"), arg("other")), "Creates a GMMStats from another GMMStats, using the copy constructor."))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::GMMStats::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this GMMStats with the 'other' one to be approximately the same.")
    .def_readwrite("log_likelihood", &bob::learn::misc::GMMStats::log_likelihood, "The accumulated log likelihood of all samples")
    .def_readwrite("t", &bob::learn::misc::GMMStats::T, "The accumulated number of samples")
    .add_property("n", &py_gmmstats_getN, &py_gmmstats_setN, "For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)")
    .add_property("sum_px", &py_gmmstats_getSumpx, &py_gmmstats_setSumpx, "For each Gaussian, the accumulated sum of responsibility times the sample ")
    .add_property("sum_pxx", &py_gmmstats_getSumpxx, &py_gmmstats_setSumpxx, "For each Gaussian, the accumulated sum of responsibility times the sample squared")
    .def("resize", &bob::learn::misc::GMMStats::resize, (arg("self"), arg("n_gaussians"), arg("n_inputs")),
         " Allocates space for the statistics and resets to zero.")
    .def("init", &bob::learn::misc::GMMStats::init, (arg("self")), "Resets statistics to zero.")
    .def("save", &s_save, (arg("self"), arg("config")), "Save to a Configuration")
    .def("load", &s_load, (arg("self"), arg("config")), "Load from a Configuration")
    .def(self_ns::str(self_ns::self))
    .def(self_ns::self += self_ns::self)
  ;

  class_<bob::learn::misc::GMMMachine, boost::shared_ptr<bob::learn::misc::GMMMachine>, bases<bob::learn::misc::Machine<blitz::Array<double,1>, double> > >("GMMMachine",
      "This class implements a multivariate diagonal Gaussian distribution.\n"
      "See Section 2.3.9 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<>(arg("self")))
    .def("__init__", boost::python::make_constructor(&m_init))
    .def(init<bob::learn::misc::GMMMachine&>((arg("self"), arg("other")), "Creates a GMMMachine from another GMMMachine, using the copy constructor."))
    .def(init<const size_t, const size_t>((arg("self"), arg("n_gaussians"), arg("n_inputs"))))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::GMMMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this GMMMachine with the 'other' one to be approximately the same.")
    .add_property("dim_d", &bob::learn::misc::GMMMachine::getNInputs, &bob::learn::misc::GMMMachine::setNInputs, "The feature dimensionality D")
    .add_property("dim_c", &bob::learn::misc::GMMMachine::getNGaussians, "The number of Gaussian components C")
    .add_property("weights", make_function(&bob::learn::misc::GMMMachine::getWeights, return_value_policy<copy_const_reference>()), &py_gmmmachine_setWeights, "The weights (also known as \"mixing coefficients\")")
    .add_property("means", &py_gmmmachine_getMeans, &py_gmmmachine_setMeans, "The means of the gaussians")
    .add_property("mean_supervector", make_function((const blitz::Array<double,1>& (bob::learn::misc::GMMMachine::*)(void) const)&bob::learn::misc::GMMMachine::getMeanSupervector, return_value_policy<copy_const_reference>()), &py_gmmmachine_setMeanSupervector,
                  "The mean supervector of the GMMMachine "
                  "(concatenation of the mean vectors of each Gaussian of the GMMMachine")
    .add_property("variances", &py_gmmmachine_getVariances, &py_gmmmachine_setVariances, "The (diagonal) variances of the Gaussians")
    .add_property("variance_supervector", make_function((const blitz::Array<double,1>& (bob::learn::misc::GMMMachine::*)(void) const)&bob::learn::misc::GMMMachine::getVarianceSupervector, return_value_policy<copy_const_reference>()), &py_gmmmachine_setVarianceSupervector,
                  "The variance supervector of the GMMMachine "
                  "(concatenation of the variance vectors of each Gaussian of the GMMMachine")
    .add_property("variance_thresholds", &py_gmmmachine_getVarianceThresholds, &py_gmmmachine_setVarianceThresholds,
                  "The variance flooring thresholds for each Gaussian in each dimension")
    .add_property("shape", &py_gmmmachine_get_shape, &py_gmmmachine_set_shape, "A tuple that represents the dimensionality of the GMMMachine ``(n_gaussians, n_inputs)``.")
    .def("resize", &bob::learn::misc::GMMMachine::resize, (arg("self"), arg("n_gaussians"), arg("n_inputs")),
         "Reset the input dimensionality, and the number of Gaussian components.\n"
         "Initialises the weights to uniform distribution.")
    .def("set_variance_thresholds", &py_gmmmachine_setVarianceThresholdsOther, (arg("self"), arg("variance_threshold")),
         "Set the variance flooring thresholds in each dimension to the same vector for all Gaussian components if the argument is a 1D numpy arrray, and equal for all Gaussian components and dimensions if the parameter is a scalar.")
    .def("update_gaussian", &bob::learn::misc::GMMMachine::updateGaussian, (arg("self"), arg("i")),
         "Get the specified Gaussian component. An exception is thrown if i is out of range.")

    .def("log_likelihood", &py_gmmmachine_loglikelihoodA, args("self", "x", "log_weighted_gaussian_likelihoods"),
         "Output the log likelihood of the sample, x, i.e. log(p(x|bob::learn::misc::GMMMachine)). Inputs are checked.")
    .def("log_likelihood_", &py_gmmmachine_loglikelihoodA_, args("self", "x", "log_weighted_gaussian_likelihoods"),
         "Output the log likelihood of the sample, x, i.e. log(p(x|bob::learn::misc::GMMMachine)). Inputs are NOT checked.")
    .def("log_likelihood", &py_gmmmachine_loglikelihoodB, args("self", "x"),
         " Output the log likelihood of the sample, x, i.e. log(p(x|GMM)). Inputs are checked.")
    .def("log_likelihood_", &py_gmmmachine_loglikelihoodB_, args("self", "x"),
         " Output the log likelihood of the sample, x, i.e. log(p(x|GMM)). Inputs are checked.")
    .def("acc_statistics", &py_gmmmachine_accStatistics, args("self", "x", "stats"),
         "Accumulate the GMM statistics for this sample(s). Inputs are checked.")
    .def("acc_statistics_", &py_gmmmachine_accStatistics_, args("self", "x", "stats"),
         "Accumulate the GMM statistics for this sample(s). Inputs are NOT checked.")
    .def("load", &m_load, (arg("self"), arg("config")), "Load from a Configuration")
    .def("save", &m_save, (arg("self"), arg("config")), "Save to a Configuration")
    .def(self_ns::str(self_ns::self))
  ;

}

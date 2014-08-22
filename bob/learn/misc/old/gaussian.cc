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
#include <bob.learn.misc/Gaussian.h>


using namespace boost::python;

static void py_setMean(bob::learn::misc::Gaussian& machine,
  bob::python::const_ndarray mean)
{
  machine.setMean(mean.bz<double,1>());
}

static void py_setVariance(bob::learn::misc::Gaussian& machine,
  bob::python::const_ndarray variance)
{
  machine.setVariance(variance.bz<double,1>());
}

static void py_setVarianceThresholds(bob::learn::misc::Gaussian& machine,
  bob::python::const_ndarray varianceThresholds)
{
  machine.setVarianceThresholds(varianceThresholds.bz<double,1>());
}

static tuple get_shape(const bob::learn::misc::Gaussian& m)
{
  return make_tuple(m.getNInputs());
}

static void set_shape(bob::learn::misc::Gaussian& m,
  const blitz::TinyVector<int,1>& s)
{
  m.resize(s(0));
}

static double py_logLikelihood(const bob::learn::misc::Gaussian& machine,
  bob::python::const_ndarray input)
{
  double output;
  machine.forward(input.bz<double,1>(), output);
  return output;
}

static double py_logLikelihood_(const bob::learn::misc::Gaussian& machine,
  bob::python::const_ndarray input)
{
  double output;
  machine.forward_(input.bz<double,1>(), output);
  return output;
}


static boost::shared_ptr<bob::learn::misc::Gaussian> _init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::Gaussian>(new bob::learn::misc::Gaussian(*hdf5->f));
}

static void _load(bob::learn::misc::Gaussian& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void _save(const bob::learn::misc::Gaussian& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}


void bind_machine_gaussian()
{
  class_<bob::learn::misc::Gaussian, boost::shared_ptr<bob::learn::misc::Gaussian>, bases<bob::learn::misc::Machine<blitz::Array<double,1>, double> > >("Gaussian",
    "This class implements a multivariate diagonal Gaussian distribution.", no_init)
    .def("__init__", boost::python::make_constructor(&_init))
    .def(init<>(arg("self")))
    .def(init<const size_t>((arg("self"), arg("n_inputs"))))
    .def(init<bob::learn::misc::Gaussian&>((arg("self"), arg("other"))))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::Gaussian::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this Gaussian with the 'other' one to be approximately the same.")
    .add_property("dim_d", &bob::learn::misc::Gaussian::getNInputs, &bob::learn::misc::Gaussian::setNInputs,
      "Dimensionality of the input feature space")
    .add_property("mean", make_function(&bob::learn::misc::Gaussian::getMean, return_value_policy<copy_const_reference>()), &py_setMean, "Mean of the Gaussian")
    .add_property("variance", make_function(&bob::learn::misc::Gaussian::getVariance, return_value_policy<copy_const_reference>()), &py_setVariance, "The diagonal of the (diagonal) covariance matrix")
    .add_property("variance_thresholds", make_function(&bob::learn::misc::Gaussian::getVarianceThresholds, return_value_policy<copy_const_reference>()), &py_setVarianceThresholds,
      "The variance flooring thresholds, i.e. the minimum allowed value of variance in each dimension. "
      "The variance will be set to this value if an attempt is made to set it to a smaller value.")
    .add_property("shape", &get_shape, &set_shape, "A tuple that represents the dimensionality of the Gaussian ``(dim_d,)``.")
    .def("set_variance_thresholds",  (void (bob::learn::misc::Gaussian::*)(const double))&bob::learn::misc::Gaussian::setVarianceThresholds, (arg("self"), arg("var_thd")),
         "Set the variance flooring thresholds equal to the given threshold for all the dimensions.")
    .def("resize", &bob::learn::misc::Gaussian::resize, (arg("self"), arg("dim_d")), "Set the input dimensionality, reset the mean to zero and the variance to one.")
    .def("log_likelihood", &py_logLikelihood, (arg("self"), arg("sample")), "Output the log likelihood of the sample, x. The input size is checked.")
    .def("log_likelihood_", &py_logLikelihood_, (arg("self"), arg("sample")), "Output the log likelihood of the sample, x. The input size is NOT checked.")
    .def("save", &_save, (arg("self"), arg("config")), "Save to a Configuration")
    .def("load", &_load, (arg("self"), arg("config")),"Load from a Configuration")
    .def(self_ns::str(self_ns::self))
  ;
}

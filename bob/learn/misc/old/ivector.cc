/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Sun Mar 31 18:07:00 2013 +0200
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>

#include "ndarray.h"

#include <bob.learn.misc/IVectorMachine.h>

using namespace boost::python;

static void py_iv_setT(bob::learn::misc::IVectorMachine& machine,
  bob::python::const_ndarray T)
{
  machine.setT(T.bz<double,2>());
}

static void py_iv_setSigma(bob::learn::misc::IVectorMachine& machine,
  bob::python::const_ndarray sigma)
{
  machine.setSigma(sigma.bz<double,1>());
}

static void py_computeIdTtSigmaInvT1(const bob::learn::misc::IVectorMachine& machine,
  const bob::learn::misc::GMMStats& gs, bob::python::ndarray output)
{
  blitz::Array<double,2> output_ = output.bz<double,2>();
  machine.computeIdTtSigmaInvT(gs, output_);
}

static object py_computeIdTtSigmaInvT2(const bob::learn::misc::IVectorMachine& machine,
  const bob::learn::misc::GMMStats& gs)
{
  bob::python::ndarray output(bob::io::base::array::t_float64, machine.getDimRt(), machine.getDimRt());
  blitz::Array<double,2> output_ = output.bz<double,2>();
  machine.computeIdTtSigmaInvT(gs, output_);
  return output.self();
}

static void py_computeTtSigmaInvFnorm1(const bob::learn::misc::IVectorMachine& machine,
  const bob::learn::misc::GMMStats& gs, bob::python::ndarray output)
{
  blitz::Array<double,1> output_ = output.bz<double,1>();
  machine.computeTtSigmaInvFnorm(gs, output_);
}

static object py_computeTtSigmaInvFnorm2(const bob::learn::misc::IVectorMachine& machine,
  const bob::learn::misc::GMMStats& gs)
{
  bob::python::ndarray output(bob::io::base::array::t_float64, machine.getDimRt());
  blitz::Array<double,1> output_ = output.bz<double,1>();
  machine.computeTtSigmaInvFnorm(gs, output_);
  return output.self();
}

static void py_iv_forward1(const bob::learn::misc::IVectorMachine& machine,
  const bob::learn::misc::GMMStats& gs, bob::python::ndarray ivector)
{
  blitz::Array<double,1> ivector_ = ivector.bz<double,1>();
  machine.forward(gs, ivector_);
}

static void py_iv_forward1_(const bob::learn::misc::IVectorMachine& machine,
  const bob::learn::misc::GMMStats& gs, bob::python::ndarray ivector)
{
  blitz::Array<double,1> ivector_ = ivector.bz<double,1>();
  machine.forward_(gs, ivector_);
}

static object py_iv_forward2(const bob::learn::misc::IVectorMachine& machine,
  const bob::learn::misc::GMMStats& gs)
{
  bob::python::ndarray ivector(bob::io::base::array::t_float64, machine.getDimRt());
  blitz::Array<double,1> ivector_ = ivector.bz<double,1>();
  machine.forward(gs, ivector_);
  return ivector.self();
}



static boost::shared_ptr<bob::learn::misc::IVectorMachine> _init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::IVectorMachine>(new bob::learn::misc::IVectorMachine(*hdf5->f));
}

static void _load(bob::learn::misc::IVectorMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void _save(const bob::learn::misc::IVectorMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}

void bind_machine_ivector()
{
  // TODO: reuse binding from generic machine
  class_<bob::learn::misc::IVectorMachine, boost::shared_ptr<bob::learn::misc::IVectorMachine> >("IVectorMachine", "An IVectorMachine to extract i-vector.\n\nReferences:\n[1] 'Front End Factor Analysis for Speaker Verification', N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, P. Ouellet, IEEE Transactions on Audio, Speech and Language Processing, 2010, vol. 19, issue 4, pp. 788-798", init<boost::shared_ptr<bob::learn::misc::GMMMachine>, optional<const size_t, const size_t> >((arg("self"), arg("ubm"), arg("rt")=1, arg("variance_threshold")=1e-10), "Builds a new IVectorMachine."))
    .def(init<>((arg("self")), "Constructs a new empty IVectorMachine."))
    .def("__init__", boost::python::make_constructor(&_init), "Constructs a new IVectorMachine from a configuration file.")
    .def(init<const bob::learn::misc::IVectorMachine&>((arg("self"), arg("machine")), "Copy constructs an IVectorMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::IVectorMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this IVectorMachine with the 'other' one to be approximately the same.")
    .def("load", &_load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &_save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::learn::misc::IVectorMachine::resize, (arg("self"), arg("rt")), "Reset the dimensionality of the Total Variability subspace T.")
    .add_property("ubm", &bob::learn::misc::IVectorMachine::getUbm, &bob::learn::misc::IVectorMachine::setUbm, "The UBM GMM attached to this Joint Factor Analysis model")
    .add_property("t", make_function(&bob::learn::misc::IVectorMachine::getT, return_value_policy<copy_const_reference>()), &py_iv_setT, "The subspace T (Total Variability matrix)")
    .add_property("sigma", make_function(&bob::learn::misc::IVectorMachine::getSigma, return_value_policy<copy_const_reference>()), &py_iv_setSigma, "The residual matrix of the model sigma")
    .add_property("variance_threshold", &bob::learn::misc::IVectorMachine::getVarianceThreshold, &bob::learn::misc::IVectorMachine::setVarianceThreshold, "Threshold for the variance contained in sigma")
    .add_property("dim_c", &bob::learn::misc::IVectorMachine::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::learn::misc::IVectorMachine::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::learn::misc::IVectorMachine::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_rt", &bob::learn::misc::IVectorMachine::getDimRt, "The dimensionality of the Total Variability subspace (rank of T)")
    .def("__compute_Id_TtSigmaInvT__", &py_computeIdTtSigmaInvT1, (arg("self"), arg("gmmstats"), arg("output")), "Computes (Id + sum_{c=1}^{C} N_{i,j,c} T^{T} Sigma_{c}^{-1} T)")
    .def("__compute_Id_TtSigmaInvT__", &py_computeIdTtSigmaInvT2, (arg("self"), arg("gmmstats")), "Computes (Id + sum_{c=1}^{C} N_{i,j,c} T^{T} Sigma_{c}^{-1} T)")
    .def("__compute_TtSigmaInvFnorm__", &py_computeTtSigmaInvFnorm1, (arg("self"), arg("gmmstats"), arg("output")), "Computes T^{T} Sigma^{-1} sum_{c=1}^{C} (F_c - N_c mean(c))")
    .def("__compute_TtSigmaInvFnorm__", &py_computeTtSigmaInvFnorm2, (arg("self"), arg("gmmstats")), "Computes T^{T} Sigma^{-1} sum_{c=1}^{C} (F_c - N_c mean(c))")
    .def("__call__", &py_iv_forward1_, (arg("self"), arg("gmmstats"), arg("ivector")), "Executes the machine on the GMMStats, and updates the ivector array. NO CHECK is performed.")
    .def("__call__", &py_iv_forward2, (arg("self"), arg("gmmstats")), "Executes the machine on the GMMStats. The ivector is allocated an returned.")
    .def("forward", &py_iv_forward1, (arg("self"), arg("gmmstats"), arg("ivector")), "Executes the machine on the GMMStats, and updates the ivector array.")
    .def("forward_", &py_iv_forward1_, (arg("self"), arg("gmmstats"), arg("ivector")), "Executes the machine on the GMMStats, and updates the ivector array. NO CHECK is performed.")
    .def("forward", &py_iv_forward2, (arg("self"), arg("gmmstats")), "Executes the machine on the GMMStats. The ivector is allocated an returned.")
  ;
}

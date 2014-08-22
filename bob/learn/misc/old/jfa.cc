/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Sat Jul 23 21:41:15 2011 +0200
 *
 * @brief Python bindings for the FA-related machines
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>

#include "ndarray.h"

#include <bob.learn.misc/JFAMachine.h>
#include <bob.learn.misc/GMMMachine.h>

using namespace boost::python;

static void py_jfa_setU(bob::learn::misc::JFABase& machine,
  bob::python::const_ndarray U)
{
  machine.setU(U.bz<double,2>());
}

static void py_jfa_setV(bob::learn::misc::JFABase& machine,
  bob::python::const_ndarray V)
{
  machine.setV(V.bz<double,2>());
}

static void py_jfa_setD(bob::learn::misc::JFABase& machine,
  bob::python::const_ndarray D)
{
  machine.setD(D.bz<double,1>());
}

static void py_jfa_setY(bob::learn::misc::JFAMachine& machine, bob::python::const_ndarray Y) {
  const blitz::Array<double,1>& Y_ = Y.bz<double,1>();
  machine.setY(Y_);
}

static void py_jfa_setZ(bob::learn::misc::JFAMachine& machine, bob::python::const_ndarray Z) {
  const blitz::Array<double,1> Z_ = Z.bz<double,1>();
  machine.setZ(Z_);
}

static void py_jfa_estimateX(bob::learn::misc::JFAMachine& machine,
  const bob::learn::misc::GMMStats& gmm_stats, bob::python::ndarray x)
{
  blitz::Array<double,1> x_ = x.bz<double,1>();
  machine.estimateX(gmm_stats, x_);
}

static void py_jfa_estimateUx(bob::learn::misc::JFAMachine& machine,
  const bob::learn::misc::GMMStats& gmm_stats, bob::python::ndarray ux)
{
  blitz::Array<double,1> ux_ = ux.bz<double,1>();
  machine.estimateUx(gmm_stats, ux_);
}

static double py_jfa_forwardUx(bob::learn::misc::JFAMachine& machine,
  const bob::learn::misc::GMMStats& gmm_stats, bob::python::const_ndarray ux)
{
  double score;
  machine.forward(gmm_stats, ux.bz<double,1>(), score);
  return score;
}


static void py_isv_setU(bob::learn::misc::ISVBase& machine,
  bob::python::const_ndarray U)
{
  machine.setU(U.bz<double,2>());
}

static void py_isv_setD(bob::learn::misc::ISVBase& machine,
  bob::python::const_ndarray D)
{
  machine.setD(D.bz<double,1>());
}

static void py_isv_setZ(bob::learn::misc::ISVMachine& machine, bob::python::const_ndarray Z) {
  machine.setZ(Z.bz<double,1>());
}

static void py_isv_estimateX(bob::learn::misc::ISVMachine& machine,
  const bob::learn::misc::GMMStats& gmm_stats, bob::python::ndarray x)
{
  blitz::Array<double,1> x_ = x.bz<double,1>();
  machine.estimateX(gmm_stats, x_);
}

static void py_isv_estimateUx(bob::learn::misc::ISVMachine& machine,
  const bob::learn::misc::GMMStats& gmm_stats, bob::python::ndarray ux)
{
  blitz::Array<double,1> ux_ = ux.bz<double,1>();
  machine.estimateUx(gmm_stats, ux_);
}

static double py_isv_forwardUx(bob::learn::misc::ISVMachine& machine,
  const bob::learn::misc::GMMStats& gmm_stats, bob::python::const_ndarray ux)
{
  double score;
  machine.forward(gmm_stats, ux.bz<double,1>(), score);
  return score;
}


static double py_gen1_forward(const bob::learn::misc::Machine<bob::learn::misc::GMMStats, double>& m,
  const bob::learn::misc::GMMStats& stats)
{
  double output;
  m.forward(stats, output);
  return output;
}

static double py_gen1_forward_(const bob::learn::misc::Machine<bob::learn::misc::GMMStats, double>& m,
  const bob::learn::misc::GMMStats& stats)
{
  double output;
  m.forward_(stats, output);
  return output;
}

static void py_gen2b_forward(const bob::learn::misc::Machine<bob::learn::misc::GMMStats, blitz::Array<double,1> >& m,
  const bob::learn::misc::GMMStats& stats, bob::python::const_ndarray output)
{
  blitz::Array<double,1> output_ = output.bz<double,1>();
  m.forward(stats, output_);
}

static void py_gen2b_forward_(const bob::learn::misc::Machine<bob::learn::misc::GMMStats, blitz::Array<double,1> >& m,
  const bob::learn::misc::GMMStats& stats, bob::python::const_ndarray output)
{
  blitz::Array<double,1> output_ = output.bz<double,1>();
  m.forward_(stats, output_);
}


static boost::shared_ptr<bob::learn::misc::JFABase> jb_init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::JFABase>(new bob::learn::misc::JFABase(*hdf5->f));
}

static void jb_load(bob::learn::misc::JFABase& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void jb_save(const bob::learn::misc::JFABase& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}


static boost::shared_ptr<bob::learn::misc::JFAMachine> jm_init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::JFAMachine>(new bob::learn::misc::JFAMachine(*hdf5->f));
}

static void jm_load(bob::learn::misc::JFAMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void jm_save(const bob::learn::misc::JFAMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}


static boost::shared_ptr<bob::learn::misc::ISVBase> ib_init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::ISVBase>(new bob::learn::misc::ISVBase(*hdf5->f));
}

static void ib_load(bob::learn::misc::ISVBase& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void ib_save(const bob::learn::misc::ISVBase& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}


static boost::shared_ptr<bob::learn::misc::ISVMachine> im_init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::ISVMachine>(new bob::learn::misc::ISVMachine(*hdf5->f));
}

static void im_load(bob::learn::misc::ISVMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void im_save(const bob::learn::misc::ISVMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}

void bind_machine_jfa()
{
  class_<bob::learn::misc::Machine<bob::learn::misc::GMMStats, double>, boost::noncopyable>("MachineGMMStatsScalarBase",
      "Root class for all Machine<bob::learn::misc::GMMStats, double>", no_init)
    .def("__call__", &py_gen1_forward_, (arg("self"), arg("input")), "Executes the machine on the GMMStats, and returns the (scalar) output. NO CHECK is performed.")
    .def("forward", &py_gen1_forward, (arg("self"), arg("input")), "Executes the machine on the GMMStats, and returns the (scalar) output.")
    .def("forward_", &py_gen1_forward_, (arg("self"), arg("input")), "Executes the machine on the GMMStats, and returns the (scalar) output. NO CHECK is performed.")
  ;

  class_<bob::learn::misc::Machine<bob::learn::misc::GMMStats, blitz::Array<double,1> >, boost::noncopyable>("MachineGMMStatsA1DBase",
      "Root class for all Machine<bob::learn::misc::GMMStats, blitz::Array<double,1>", no_init)
    .def("__call__", &py_gen2b_forward_, (arg("self"), arg("input"), arg("output")), "Executes the machine on the GMMStats, and returns the (scalar) output. NO CHECK is performed.")
    .def("forward", &py_gen2b_forward, (arg("self"), arg("input"), arg("output")), "Executes the machine on the GMMStats, and returns the (scalar) output.")
    .def("forward_", &py_gen2b_forward_, (arg("self"), arg("input"), arg("output")), "Executes the machine on the GMMStats, and returns the (scalar) output. NO CHECK is performed.")
  ;


  class_<bob::learn::misc::JFABase, boost::shared_ptr<bob::learn::misc::JFABase>, bases<bob::learn::misc::Machine<bob::learn::misc::GMMStats, double> > >("JFABase", "A JFABase instance can be seen as a container for U, V and D when performing Joint Factor Analysis (JFA).\n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", no_init)
    .def("__init__", boost::python::make_constructor(&jb_init), "Constructs a new JFABaseMachine from a configuration file.")
    .def(init<const boost::shared_ptr<bob::learn::misc::GMMMachine>, optional<const size_t, const size_t> >((arg("self"), arg("ubm"), arg("ru")=1, arg("rv")=1), "Builds a new JFABase."))
    .def(init<>((arg("self")), "Constructs a 1x1 JFABase instance. You have to set a UBM GMM and resize the U, V and D subspaces afterwards."))
    .def(init<const bob::learn::misc::JFABase&>((arg("self"), arg("machine")), "Copy constructs a JFABase"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::JFABase::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this JFABase with the 'other' one to be approximately the same.")
    .def("load", &jb_load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &jb_save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::learn::misc::JFABase::resize, (arg("self"), arg("ru"), arg("rv")), "Reset the dimensionality of the subspaces U and V.")
    .add_property("ubm", &bob::learn::misc::JFABase::getUbm, &bob::learn::misc::JFABase::setUbm, "The UBM GMM attached to this Joint Factor Analysis model")
    .add_property("u", make_function(&bob::learn::misc::JFABase::getU, return_value_policy<copy_const_reference>()), &py_jfa_setU, "The subspace U for within-class variations")
    .add_property("v", make_function(&bob::learn::misc::JFABase::getV, return_value_policy<copy_const_reference>()), &py_jfa_setV, "The subspace V for between-class variations")
    .add_property("d", make_function(&bob::learn::misc::JFABase::getD, return_value_policy<copy_const_reference>()), &py_jfa_setD, "The subspace D for residual variations")
    .add_property("dim_c", &bob::learn::misc::JFABase::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::learn::misc::JFABase::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::learn::misc::JFABase::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_ru", &bob::learn::misc::JFABase::getDimRu, "The dimensionality of the within-class variations subspace (rank of U)")
    .add_property("dim_rv", &bob::learn::misc::JFABase::getDimRv, "The dimensionality of the between-class variations subspace (rank of V)")
  ;

  class_<bob::learn::misc::JFAMachine, boost::shared_ptr<bob::learn::misc::JFAMachine>, bases<bob::learn::misc::Machine<bob::learn::misc::GMMStats, double> > >("JFAMachine", "A JFAMachine. An attached JFABase should be provided for Joint Factor Analysis. The JFAMachine carries information about the speaker factors y and z, whereas a JFABase carries information about the matrices U, V and D.\n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", no_init)
    .def("__init__", boost::python::make_constructor(&jm_init), "Constructs a new JFAMachine from a configuration file.")
    .def(init<>((arg("self")), "Constructs a 1x1 JFAMachine instance. You have to set a JFABase afterwards."))
    .def(init<const boost::shared_ptr<bob::learn::misc::JFABase> >((arg("self"), arg("jfa_base")), "Builds a new JFAMachine."))
    .def(init<const bob::learn::misc::JFAMachine&>((arg("self"), arg("machine")), "Copy constructs a JFAMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::JFAMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this JFABase with the 'other' one to be approximately the same.")
    .def("load", &jm_load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &jm_save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("estimate_x", &py_jfa_estimateX, (arg("self"), arg("stats"), arg("x")), "Estimates the session offset x (LPT assumption) given GMM statistics.")
    .def("estimate_ux", &py_jfa_estimateUx, (arg("self"), arg("stats"), arg("ux")), "Estimates Ux (LPT assumption) given GMM statistics.")
    .def("forward_ux", &py_jfa_forwardUx, (arg("self"), arg("stats"), arg("ux")), "Processes the GMM statistics and Ux to return a score.")
    .add_property("jfa_base", &bob::learn::misc::JFAMachine::getJFABase, &bob::learn::misc::JFAMachine::setJFABase, "The JFABase attached to this machine")
    .add_property("__x__", make_function(&bob::learn::misc::JFAMachine::getX, return_value_policy<copy_const_reference>()), "The latent variable x (last one computed). This is a feature provided for convenience, but this attribute is not 'part' of the machine. The session latent variable x is indeed not class-specific, but depends on the sample considered. Furthermore, it is not saved into the machine or used when comparing machines.")
    .add_property("y", make_function(&bob::learn::misc::JFAMachine::getY, return_value_policy<copy_const_reference>()), &py_jfa_setY, "The latent variable y of this machine")
    .add_property("z", make_function(&bob::learn::misc::JFAMachine::getZ, return_value_policy<copy_const_reference>()), &py_jfa_setZ, "The latent variable z of this machine")
    .add_property("dim_c", &bob::learn::misc::JFAMachine::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::learn::misc::JFAMachine::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::learn::misc::JFAMachine::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_ru", &bob::learn::misc::JFAMachine::getDimRu, "The dimensionality of the within-class variations subspace (rank of U)")
    .add_property("dim_rv", &bob::learn::misc::JFAMachine::getDimRv, "The dimensionality of the between-class variations subspace (rank of V)")
  ;

  class_<bob::learn::misc::ISVBase, boost::shared_ptr<bob::learn::misc::ISVBase>, bases<bob::learn::misc::Machine<bob::learn::misc::GMMStats, double> > >("ISVBase", "An ISVBase instance can be seen as a container for U and D when performing Joint Factor Analysis (ISV). \n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", no_init)
    .def("__init__", boost::python::make_constructor(&ib_init), "Constructs a new ISVBaseMachine from a configuration file.")
    .def(init<const boost::shared_ptr<bob::learn::misc::GMMMachine>, optional<const size_t> >((arg("self"), arg("ubm"), arg("ru")=1), "Builds a new ISVBase."))
    .def(init<>((arg("self")), "Constructs a 1 ISVBase instance. You have to set a UBM GMM and resize the U and D subspaces afterwards."))
    .def(init<const bob::learn::misc::ISVBase&>((arg("self"), arg("machine")), "Copy constructs an ISVBase"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::ISVBase::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this ISVBase with the 'other' one to be approximately the same.")
    .def("load", &ib_load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &ib_save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::learn::misc::ISVBase::resize, (arg("self"), arg("ru")), "Reset the dimensionality of the subspaces U.")
    .add_property("ubm", &bob::learn::misc::ISVBase::getUbm, &bob::learn::misc::ISVBase::setUbm, "The UBM GMM attached to this Joint Factor Analysis model")
    .add_property("u", make_function(&bob::learn::misc::ISVBase::getU, return_value_policy<copy_const_reference>()), &py_isv_setU, "The subspace U for within-class variations")
    .add_property("d", make_function(&bob::learn::misc::ISVBase::getD, return_value_policy<copy_const_reference>()), &py_isv_setD, "The subspace D for residual variations")
    .add_property("dim_c", &bob::learn::misc::ISVBase::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::learn::misc::ISVBase::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::learn::misc::ISVBase::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_ru", &bob::learn::misc::ISVBase::getDimRu, "The dimensionality of the within-class variations subspace (rank of U)")
  ;

  class_<bob::learn::misc::ISVMachine, boost::shared_ptr<bob::learn::misc::ISVMachine>, bases<bob::learn::misc::Machine<bob::learn::misc::GMMStats, double> > >("ISVMachine", "An ISVMachine. An attached ISVBase should be provided for Inter-session Variability Modelling. The ISVMachine carries information about the speaker factors z, whereas a ISVBase carries information about the matrices U and D. \n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", no_init)
    .def("__init__", boost::python::make_constructor(&im_init), "Constructs a new ISVMachine from a configuration file.")
    .def(init<>((arg("self")), "Constructs a 1 ISVMachine instance. You have to set a ISVBase afterwards."))
    .def(init<const boost::shared_ptr<bob::learn::misc::ISVBase> >((arg("self"), arg("isv_base")), "Builds a new ISVMachine."))
    .def(init<const bob::learn::misc::ISVMachine&>((arg("self"), arg("machine")), "Copy constructs an ISVMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::ISVMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this ISVBase with the 'other' one to be approximately the same.")
    .def("load", &im_load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &im_save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("estimate_x", &py_isv_estimateX, (arg("self"), arg("stats"), arg("x")), "Estimates the session offset x (LPT assumption) given GMM statistics.")
    .def("estimate_ux", &py_isv_estimateUx, (arg("self"), arg("stats"), arg("ux")), "Estimates Ux (LPT assumption) given GMM statistics.")
    .def("forward_ux", &py_isv_forwardUx, (arg("self"), arg("stats"), arg("ux")), "Processes the GMM statistics and Ux to return a score.")
    .add_property("isv_base", &bob::learn::misc::ISVMachine::getISVBase, &bob::learn::misc::ISVMachine::setISVBase, "The ISVBase attached to this machine")
    .add_property("__x__", make_function(&bob::learn::misc::ISVMachine::getX, return_value_policy<copy_const_reference>()), "The latent variable x (last one computed). This is a feature provided for convenience, but this attribute is not 'part' of the machine. The session latent variable x is indeed not class-specific, but depends on the sample considered. Furthermore, it is not saved into the machine or used when comparing machines.")
    .add_property("z", make_function(&bob::learn::misc::ISVMachine::getZ, return_value_policy<copy_const_reference>()), &py_isv_setZ, "The latent variable z of this machine")
    .add_property("dim_c", &bob::learn::misc::ISVMachine::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::learn::misc::ISVMachine::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::learn::misc::ISVMachine::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_ru", &bob::learn::misc::ISVMachine::getDimRu, "The dimensionality of the within-class variations subspace (rank of U)")
  ;
}

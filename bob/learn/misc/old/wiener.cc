/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Fri Sep 30 16:58:42 2011 +0200
 *
 * @brief Bindings for a WienerMachine
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>

#include "ndarray.h"
#include <bob.learn.misc/WienerMachine.h>

using namespace boost::python;

static void py_forward1_(const bob::learn::misc::WienerMachine& m,
  bob::python::const_ndarray input, bob::python::ndarray output)
{
  blitz::Array<double,2> output_ = output.bz<double,2>();
  m.forward_(input.bz<double,2>(), output_);
}

static void py_forward1(const bob::learn::misc::WienerMachine& m,
  bob::python::const_ndarray input, bob::python::ndarray output)
{
  blitz::Array<double,2> output_ = output.bz<double,2>();
  m.forward(input.bz<double,2>(), output_);
}

static object py_forward2(const bob::learn::misc::WienerMachine& m,
  bob::python::const_ndarray input)
{
  const bob::io::base::array::typeinfo& info = input.type();
  bob::python::ndarray output(bob::io::base::array::t_float64, info.shape[0], info.shape[1]);
  blitz::Array<double,2> output_ = output.bz<double,2>();
  m.forward(input.bz<double,2>(), output_);
  return output.self();
}

static tuple get_shape(const bob::learn::misc::WienerMachine& m)
{
  return make_tuple(m.getHeight(), m.getWidth());
}

static void set_shape(bob::learn::misc::WienerMachine& m,
    const blitz::TinyVector<int,2>& s)
{
  m.resize(s(0), s(1));
}

static void py_set_ps(bob::learn::misc::WienerMachine& m,
  bob::python::const_ndarray ps)
{
  m.setPs(ps.bz<double,2>());
}


static boost::shared_ptr<bob::learn::misc::WienerMachine> _init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::WienerMachine>(new bob::learn::misc::WienerMachine(*hdf5->f));
}

static void _load(bob::learn::misc::WienerMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void _save(const bob::learn::misc::WienerMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}


void bind_machine_wiener()
{
  class_<bob::learn::misc::WienerMachine, boost::shared_ptr<bob::learn::misc::WienerMachine> >("WienerMachine", "A Wiener filter.\nReference:\n'Computer Vision: Algorithms and Applications', Richard Szeliski, (Part 3.4.3)", init<const size_t, const size_t, const double, optional<const double> >((arg("self"), arg("height"), arg("width"), arg("pn"), arg("variance_threshold")=1e-8), "Constructs a new Wiener filter dedicated to images of the given dimensions. The filter is initialized with zero values."))
    .def(init<const blitz::Array<double,2>&, const double> ((arg("self"), arg("ps"), arg("pn")), "Constructs a new WienerMachine from a set of variance estimates ps, a noise level pn."))
    .def(init<>((arg("self")), "Default constructor, builds a machine as with 'WienerMachine(0,0,0)'."))
    .def("__init__", boost::python::make_constructor(&_init), "Constructs a new WienerMachine from a configuration file.")
    .def(init<const bob::learn::misc::WienerMachine&>((arg("self"), arg("machine")), "Copy constructs an WienerMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::WienerMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this WienerMachine with the 'other' one to be approximately the same.")
    .def("load", &_load, (arg("self"), arg("config")), "Loads the filter from a configuration file.")
    .def("save", &_save, (arg("self"), arg("config")), "Saves the filter to a configuration file.")
    .add_property("pn", &bob::learn::misc::WienerMachine::getPn, &bob::learn::misc::WienerMachine::setPn, "Noise level Pn")
    .add_property("variance_threshold", &bob::learn::misc::WienerMachine::getVarianceThreshold, &bob::learn::misc::WienerMachine::setVarianceThreshold, "Variance flooring threshold (min variance value)")
    .add_property("ps",make_function(&bob::learn::misc::WienerMachine::getPs, return_value_policy<copy_const_reference>()), &py_set_ps, "Variance Ps estimated at each frequency")
    .add_property("w", make_function(&bob::learn::misc::WienerMachine::getW, return_value_policy<copy_const_reference>()), "The Wiener filter W (W=1/(1+Pn/Ps)) (read-only)")
    .add_property("height", &bob::learn::misc::WienerMachine::getHeight, &bob::learn::misc::WienerMachine::setHeight, "Height of the filter/image to process")
    .add_property("width", &bob::learn::misc::WienerMachine::getWidth, &bob::learn::misc::WienerMachine::setWidth, "Width of the filter/image to process")
    .add_property("shape", &get_shape, &set_shape)
    .def("__call__", &py_forward1, (arg("self"), arg("input"), arg("output")), "Filters the input and saves results on the output.")
    .def("forward", &py_forward1, (arg("self"), arg("input"), arg("output")), "Filters the input and saves results on the output.")
    .def("forward_", &py_forward1_, (arg("self"), arg("input"), arg("output")), "Filters the input and saves results on the output. Input is not checked.")
    .def("__call__", &py_forward2, (arg("self"), arg("input")), "Filters the input and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    .def("forward", &py_forward2, (arg("self"), arg("input")), "Filter the input and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    ;
}

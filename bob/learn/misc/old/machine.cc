/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue Jul 26 15:11:33 2011 +0200
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "ndarray.h"
#include <bob.learn.misc/Machine.h>

using namespace boost::python;

static double forward(const bob::learn::misc::Machine<blitz::Array<double,1>, double>& m,
    bob::python::const_ndarray input) {
  double output;
  m.forward(input.bz<double,1>(), output);
  return output;
}

static double forward_(const bob::learn::misc::Machine<blitz::Array<double,1>, double>& m,
    bob::python::const_ndarray input) {
  double output;
  m.forward_(input.bz<double,1>(), output);
  return output;
}

void bind_machine_base()
{
  class_<bob::learn::misc::Machine<blitz::Array<double,1>, double>, boost::noncopyable>("MachineDoubleBase",
      "Root class for all Machine<blitz::Array<double,1>, double>", no_init)
    .def("__call__", &forward_, (arg("self"), arg("input")), "Executes the machine on the given 1D numpy array of float64, and returns the output. NO CHECK is performed.")
    .def("forward", &forward, (arg("self"), arg("input")), "Executes the machine on the given 1D numpy array of float64, and returns the output.")
    .def("forward_", &forward_, (arg("self"), arg("input")), "Executes the machine on the given 1D numpy array of float64, and returns the output. NO CHECK is performed.")
  ;
}

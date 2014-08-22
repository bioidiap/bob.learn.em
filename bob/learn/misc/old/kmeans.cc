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

#include <bob.learn.misc/KMeansMachine.h>

using namespace boost::python;

static tuple py_getVariancesAndWeightsForEachCluster(const bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray ar) {
  size_t n_means = machine.getNMeans();
  size_t n_inputs = machine.getNInputs();
  bob::python::ndarray variances(bob::io::base::array::t_float64, n_means, n_inputs);
  bob::python::ndarray weights(bob::io::base::array::t_float64, n_means);
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  machine.getVariancesAndWeightsForEachCluster(ar.bz<double,2>(), variances_, weights_);
  return boost::python::make_tuple(variances.self(), weights.self());
}

static void py_getVariancesAndWeightsForEachClusterInit(const bob::learn::misc::KMeansMachine& machine, bob::python::ndarray variances, bob::python::ndarray weights) {
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  machine.getVariancesAndWeightsForEachClusterInit(variances_, weights_);
}

static void py_getVariancesAndWeightsForEachClusterAcc(const bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray ar, bob::python::ndarray variances, bob::python::ndarray weights) {
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  machine.getVariancesAndWeightsForEachClusterAcc(ar.bz<double,2>(), variances_, weights_);
}

static void py_getVariancesAndWeightsForEachClusterFin(const bob::learn::misc::KMeansMachine& machine, bob::python::ndarray variances, bob::python::ndarray weights) {
  blitz::Array<double,2> variances_ = variances.bz<double,2>();
  blitz::Array<double,1> weights_ = weights.bz<double,1>();
  machine.getVariancesAndWeightsForEachClusterFin(variances_, weights_);
}

static object py_getMean(const bob::learn::misc::KMeansMachine& kMeansMachine, const size_t i) {
  bob::python::ndarray mean(bob::io::base::array::t_float64, kMeansMachine.getNInputs());
  blitz::Array<double,1> mean_ = mean.bz<double,1>();
  kMeansMachine.getMean(i, mean_);
  return mean.self();
}

static void py_setMean(bob::learn::misc::KMeansMachine& machine, const size_t i, bob::python::const_ndarray mean) {
  machine.setMean(i, mean.bz<double,1>());
}

static void py_setMeans(bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray means) {
  machine.setMeans(means.bz<double,2>());
}

static double py_getDistanceFromMean(const bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray x, const size_t i)
{
  return machine.getDistanceFromMean(x.bz<double,1>(), i);
}

static tuple py_getClosestMean(const bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray x)
{
  size_t closest_mean;
  double min_distance;
  machine.getClosestMean(x.bz<double,1>(), closest_mean, min_distance);
  return boost::python::make_tuple(closest_mean, min_distance);
}

static double py_getMinDistance(const bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray input)
{
  return machine.getMinDistance(input.bz<double,1>());
}

static void py_setCacheMeans(bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray cache_means) {
  machine.setCacheMeans(cache_means.bz<double,2>());
}


static boost::shared_ptr<bob::learn::misc::KMeansMachine> _init(boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  return boost::shared_ptr<bob::learn::misc::KMeansMachine>(new bob::learn::misc::KMeansMachine(*hdf5->f));
}

static void _load(bob::learn::misc::KMeansMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.load(*hdf5->f);
}

static void _save(const bob::learn::misc::KMeansMachine& self, boost::python::object file){
  if (!PyBobIoHDF5File_Check(file.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.io.base.HDF5File");
  PyBobIoHDF5FileObject* hdf5 = (PyBobIoHDF5FileObject*) file.ptr();
  self.save(*hdf5->f);
}

void bind_machine_kmeans()
{
  class_<bob::learn::misc::KMeansMachine, boost::shared_ptr<bob::learn::misc::KMeansMachine>,
         bases<bob::learn::misc::Machine<blitz::Array<double,1>, double> > >("KMeansMachine",
      "This class implements a k-means classifier.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006",
      init<>((arg("self"))))
    .def("__init__", boost::python::make_constructor(&_init))
    .def(init<const size_t, const size_t>((arg("self"), arg("n_means"), arg("n_inputs"))))
    .def(init<bob::learn::misc::KMeansMachine&>((arg("self"), arg("other"))))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::KMeansMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this KMeansMachine with the 'other' one to be approximately the same.")
    .add_property("means", make_function(&bob::learn::misc::KMeansMachine::getMeans, return_value_policy<copy_const_reference>()), &py_setMeans, "The mean vectors")
    .add_property("__cache_means__", make_function(&bob::learn::misc::KMeansMachine::getCacheMeans, return_value_policy<copy_const_reference>()), &py_setCacheMeans, "The cache mean vectors. This should only be used when parallelizing the get_variances_and_weights_for_each_cluster() method")
    .add_property("dim_d", &bob::learn::misc::KMeansMachine::getNInputs, "Number of inputs")
    .add_property("dim_c", &bob::learn::misc::KMeansMachine::getNMeans, "Number of means (k)")
    .def("resize", &bob::learn::misc::KMeansMachine::resize, (arg("self"), arg("n_means"), arg("n_inputs")), "Resize the number of means and inputs")
    .def("get_mean", &py_getMean, (arg("self"), arg("i")), "Get the i'th mean")
    .def("set_mean", &py_setMean, (arg("self"), arg("i"), arg("mean")), "Set the i'th mean")
    .def("get_distance_from_mean", &py_getDistanceFromMean, (arg("self"), arg("x"), arg("i")),
        "Return the power of two of the square Euclidean distance of the sample, x, to the i'th mean")
    .def("get_closest_mean", &py_getClosestMean, (arg("self"), arg("x")),
        "Calculate the index of the mean that is closest (in terms of square Euclidean distance) to the data sample, x")
    .def("get_min_distance", &py_getMinDistance, (arg("self"), arg("input")),
        "Output the minimum square Euclidean distance between the input and one of the means")
    .def("get_variances_and_weights_for_each_cluster", &py_getVariancesAndWeightsForEachCluster, (arg("self"), arg("data")),
        "For each mean, find the subset of the samples that is closest to that mean, and calculate\n"
        "1) the variance of that subset (the cluster variance)\n"
        "2) the proportion of the samples represented by that subset (the cluster weight)")
    .def("__get_variances_and_weights_for_each_cluster_init__", &py_getVariancesAndWeightsForEachClusterInit, (arg("self"), arg("variances"), arg("weights")),
        "For the parallel version of get_variances_and_weights_for_each_cluster()\n"
        "Initialization step")
    .def("__get_variances_and_weights_for_each_cluster_acc__", &py_getVariancesAndWeightsForEachClusterAcc, (arg("self"), arg("data"), arg("variances"), arg("weights")),
        "For the parallel version of get_variances_and_weights_for_each_cluster()\n"
        "Accumulation step")
    .def("__get_variances_and_weights_for_each_cluster_fin__", &py_getVariancesAndWeightsForEachClusterFin, (arg("self"), arg("variances"), arg("weights")),
        "For the parallel version of get_variances_and_weights_for_each_cluster()\n"
        "Finalization step")
    .def("load", &_load, (arg("self"), arg("config")), "Load from a Configuration")
    .def("save", &_save, (arg("self"), arg("config")), "Save to a Configuration")
    .def(self_ns::str(self_ns::self))
  ;
}

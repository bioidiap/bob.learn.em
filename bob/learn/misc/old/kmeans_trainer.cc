/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu Jun 9 18:12:33 2011 +0200
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "ndarray.h"
#include <bob.learn.misc/KMeansTrainer.h>

using namespace boost::python;

typedef bob::learn::misc::EMTrainer<bob::learn::misc::KMeansMachine, blitz::Array<double,2> > EMTrainerKMeansBase;

static void py_setZeroethOrderStats(bob::learn::misc::KMeansTrainer& op, bob::python::const_ndarray stats) {
  const bob::io::base::array::typeinfo& info = stats.type();
  if(info.dtype != bob::io::base::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  op.setZeroethOrderStats(stats.bz<double,1>());
}

static void py_setFirstOrderStats(bob::learn::misc::KMeansTrainer& op, bob::python::const_ndarray stats) {
  const bob::io::base::array::typeinfo& info = stats.type();
  if(info.dtype != bob::io::base::array::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  op.setFirstOrderStats(stats.bz<double,2>());
}

static void py_train(EMTrainerKMeansBase& trainer,
  bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.train(machine, sample.bz<double,2>());
}

static void py_initialize(EMTrainerKMeansBase& trainer,
  bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.initialize(machine, sample.bz<double,2>());
}

static void py_finalize(EMTrainerKMeansBase& trainer,
  bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.finalize(machine, sample.bz<double,2>());
}

static void py_eStep(EMTrainerKMeansBase& trainer,
  bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.eStep(machine, sample.bz<double,2>());
}

static void py_mStep(EMTrainerKMeansBase& trainer,
  bob::learn::misc::KMeansMachine& machine, bob::python::const_ndarray sample)
{
  trainer.mStep(machine, sample.bz<double,2>());
}

// include the random API of bob.core
#include <bob.core/random_api.h>
static boost::python::object KMTB_getRng(EMTrainerKMeansBase& self){
  // create new object
  PyObject* o = PyBoostMt19937_Type.tp_alloc(&PyBoostMt19937_Type,0);
  reinterpret_cast<PyBoostMt19937Object*>(o)->rng = self.getRng().get();
  return boost::python::object(boost::python::handle<>(boost::python::borrowed(o)));
}
static boost::python::object KMT_getRng(bob::learn::misc::KMeansTrainer& self){
  // create new object
  PyObject* o = PyBoostMt19937_Type.tp_alloc(&PyBoostMt19937_Type,0);
  reinterpret_cast<PyBoostMt19937Object*>(o)->rng = self.getRng().get();
  return boost::python::object(boost::python::handle<>(boost::python::borrowed(o)));
}

#include <boost/make_shared.hpp>
static void KMTB_setRng(EMTrainerKMeansBase& self, boost::python::object rng){
  if (!PyBoostMt19937_Check(rng.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.core.random.mt19937 object");
  PyBoostMt19937Object* o = reinterpret_cast<PyBoostMt19937Object*>(rng.ptr());
  self.setRng(boost::make_shared<boost::mt19937>(*o->rng));
}
static void KMT_setRng(bob::learn::misc::KMeansTrainer& self, boost::python::object rng){
  if (!PyBoostMt19937_Check(rng.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.core.random.mt19937 object");
  PyBoostMt19937Object* o = reinterpret_cast<PyBoostMt19937Object*>(rng.ptr());
  self.setRng(boost::make_shared<boost::mt19937>(*o->rng));
}


void bind_trainer_kmeans()
{
  class_<EMTrainerKMeansBase, boost::noncopyable>("EMTrainerKMeans", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergence_threshold", &EMTrainerKMeansBase::getConvergenceThreshold, &EMTrainerKMeansBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerKMeansBase::getMaxIterations, &EMTrainerKMeansBase::setMaxIterations, "Max iterations")
    .add_property("compute_likelihood", &EMTrainerKMeansBase::getComputeLikelihood, &EMTrainerKMeansBase::setComputeLikelihood, "Tells whether we compute the average min (square Euclidean) distance or not.")
    .add_property("rng", &KMTB_getRng, &KMTB_setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .def(self == self)
    .def(self != self)
    .def("train", &py_train, (arg("self"), arg("machine"), arg("data")), "Train a machine using data")
    .def("initialize", &py_initialize, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("e_step", &py_eStep, (arg("self"), arg("machine"), arg("data")),
       "Update the hidden variable distribution (or the sufficient statistics) given the Machine parameters. "
       "Also, calculate the average output of the Machine given these parameters.\n"
       "Return the average output of the Machine across the dataset. "
       "The EM algorithm will terminate once the change in average_output "
       "is less than the convergence_threshold.")
    .def("m_step", &py_mStep, (arg("self"), arg("machine"), arg("data")), "Update the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &EMTrainerKMeansBase::computeLikelihood, (arg("self"), arg("machine")), "Returns the average min (square Euclidean) distance")
    .def("finalize", &py_finalize, (arg("self"), arg("machine"), arg("data")), "This method is called after the EM algorithm")
  ;

  // Starts binding the KMeansTrainer
  class_<bob::learn::misc::KMeansTrainer, boost::shared_ptr<bob::learn::misc::KMeansTrainer>, boost::noncopyable, bases<EMTrainerKMeansBase> > KMT("KMeansTrainer",
      "Trains a KMeans machine.\n"
      "This class implements the expectation-maximization algorithm for a k-means machine.\n"
      "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006\n"
      "It uses a random initialization of the means followed by the expectation-maximization algorithm",
      no_init
      );

  // Binds methods that does not have nested enum values as default parameters
  KMT.def(self == self)
     .def(self != self)
     .add_property("initialization_method", &bob::learn::misc::KMeansTrainer::getInitializationMethod, &bob::learn::misc::KMeansTrainer::setInitializationMethod, "The initialization method to generate the initial means.")
     .add_property("rng", &KMT_getRng, &KMT_setRng, "The Mersenne Twister mt19937 random generator used for the initialization of the means.")
     .add_property("average_min_distance", &bob::learn::misc::KMeansTrainer::getAverageMinDistance, &bob::learn::misc::KMeansTrainer::setAverageMinDistance, "Average min (square Euclidean) distance. Useful to parallelize the E-step.")
     .add_property("zeroeth_order_statistics", make_function(&bob::learn::misc::KMeansTrainer::getZeroethOrderStats, return_value_policy<copy_const_reference>()), &py_setZeroethOrderStats, "The zeroeth order statistics. Useful to parallelize the E-step.")
     .add_property("first_order_statistics", make_function(&bob::learn::misc::KMeansTrainer::getFirstOrderStats, return_value_policy<copy_const_reference>()), &py_setFirstOrderStats, "The first order statistics. Useful to parallelize the E-step.")
    ;

  // Sets the scope to the one of the KMeansTrainer
  scope s(KMT);

  // Adds enum in the previously defined current scope
  enum_<bob::learn::misc::KMeansTrainer::InitializationMethod>("initialization_method_type")
    .value("RANDOM", bob::learn::misc::KMeansTrainer::RANDOM)
    .value("RANDOM_NO_DUPLICATE", bob::learn::misc::KMeansTrainer::RANDOM_NO_DUPLICATE)
#if BOOST_VERSION >= 104700
    .value("KMEANS_PLUS_PLUS", bob::learn::misc::KMeansTrainer::KMEANS_PLUS_PLUS)
#endif
    .export_values()
    ;

  // Binds methods that has nested enum values as default parameters
  KMT.def(init<optional<double,int,bool,bob::learn::misc::KMeansTrainer::InitializationMethod> >((arg("self"), arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=true, arg("initialization_method")=bob::learn::misc::KMeansTrainer::RANDOM)));
}

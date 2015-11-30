/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Tue 13 Jan 16:50:00 2015
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

// InitializationMethod type conversion

#if BOOST_VERSION >= 104700
  static const std::map<std::string, bob::learn::em::KMeansTrainer::InitializationMethod> IM = {{"RANDOM",  bob::learn::em::KMeansTrainer::InitializationMethod::RANDOM},  {"RANDOM_NO_DUPLICATE", bob::learn::em::KMeansTrainer::InitializationMethod::RANDOM_NO_DUPLICATE}, {"KMEANS_PLUS_PLUS", bob::learn::em::KMeansTrainer::InitializationMethod::KMEANS_PLUS_PLUS}};
#else
  static const std::map<std::string, bob::learn::em::KMeansTrainer::InitializationMethod> IM = {{"RANDOM",  bob::learn::em::KMeansTrainer::InitializationMethod::RANDOM}, {"RANDOM_NO_DUPLICATE", bob::learn::em::KMeansTrainer::InitializationMethod::RANDOM_NO_DUPLICATE}};
#endif

static inline bob::learn::em::KMeansTrainer::InitializationMethod string2IM(const std::string& o){            /* converts string to InitializationMethod type */
  auto it = IM.find(o);
  if (it == IM.end()) throw std::runtime_error("The given InitializationMethod '" + o + "' is not known; choose one of ('RANDOM', 'RANDOM_NO_DUPLICATE', 'KMEANS_PLUS_PLUS')");
  else return it->second;
}
static inline const std::string& IM2string(bob::learn::em::KMeansTrainer::InitializationMethod o){            /* converts InitializationMethod type to string */
  for (auto it = IM.begin(); it != IM.end(); ++it) if (it->second == o) return it->first;
  throw std::runtime_error("The given InitializationMethod type is not known");
}


static auto KMeansTrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".KMeansTrainer",
  "Trains a KMeans machine."
  "This class implements the expectation-maximization algorithm for a k-means machine."
  "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006"
  "It uses a random initialization of the means followed by the expectation-maximization algorithm"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a KMeansTrainer",
    "",
    true
  )
  .add_prototype("initialization_method","")
  .add_prototype("other","")
  .add_prototype("","")

  .add_parameter("initialization_method", "str", "The initialization method of the means.\nPossible values are: 'RANDOM', 'RANDOM_NO_DUPLICATE', 'KMEANS_PLUS_PLUS' ")
  .add_parameter("other", ":py:class:`bob.learn.em.KMeansTrainer`", "A KMeansTrainer object to be copied.")

);


static int PyBobLearnEMKMeansTrainer_init_copy(PyBobLearnEMKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = KMeansTrainer_doc.kwlist(1);
  PyBobLearnEMKMeansTrainerObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMKMeansTrainer_Type, &tt)){
    KMeansTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::KMeansTrainer(*tt->cxx));
  return 0;
}

static int PyBobLearnEMKMeansTrainer_init_str(PyBobLearnEMKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = KMeansTrainer_doc.kwlist(0);
  char* value;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", kwlist, &value)){
    KMeansTrainer_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::learn::em::KMeansTrainer(string2IM(std::string(value))));
  return 0;
}


static int PyBobLearnEMKMeansTrainer_init(PyBobLearnEMKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs) {

    case 0:{ //default initializer ()
      self->cxx.reset(new bob::learn::em::KMeansTrainer());
      return 0;
    }
    case 1:{
      //Reading the input argument
      PyObject* arg = 0;
      if (PyTuple_Size(args))
        arg = PyTuple_GET_ITEM(args, 0);
      else {
        PyObject* tmp = PyDict_Values(kwargs);
        auto tmp_ = make_safe(tmp);
        arg = PyList_GET_ITEM(tmp, 0);
      }

      // If the constructor input is KMeansTrainer object
      if (PyBobLearnEMKMeansTrainer_Check(arg))
        return PyBobLearnEMKMeansTrainer_init_copy(self, args, kwargs);
      else if(PyString_Check(arg))
        return PyBobLearnEMKMeansTrainer_init_str(self, args, kwargs);
    }
    default:{
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0 or 1 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      KMeansTrainer_doc.print_usage();
      return -1;
    }
  }
  BOB_CATCH_MEMBER("cannot create KMeansTrainer", -1)
  return 0;
}


static void PyBobLearnEMKMeansTrainer_delete(PyBobLearnEMKMeansTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnEMKMeansTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMKMeansTrainer_Type));
}


static PyObject* PyBobLearnEMKMeansTrainer_RichCompare(PyBobLearnEMKMeansTrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMKMeansTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMKMeansTrainerObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare KMeansTrainer objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** initialization_method *****/
static auto initialization_method = bob::extension::VariableDoc(
  "initialization_method",
  "str",
  "Initialization method.",
  "Possible values:\n"
  " `RANDOM`: Random initialization \n\n"
  " `RANDOM_NO_DUPLICATE`: Random initialization without repetition \n\n"
  " `KMEANS_PLUS_PLUS`: Apply the kmeans++ initialization http://en.wikipedia.org/wiki/K-means%2B%2B  \n\n"
);
PyObject* PyBobLearnEMKMeansTrainer_getInitializationMethod(PyBobLearnEMKMeansTrainerObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("s", IM2string(self->cxx->getInitializationMethod()).c_str());
  BOB_CATCH_MEMBER("initialization method could not be read", 0)
}
int PyBobLearnEMKMeansTrainer_setInitializationMethod(PyBobLearnEMKMeansTrainerObject* self, PyObject* value, void*) {
  BOB_TRY

  if (!PyString_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an str", Py_TYPE(self)->tp_name, initialization_method.name());
    return -1;
  }
  self->cxx->setInitializationMethod(string2IM(PyString_AS_STRING(value)));

  return 0;
  BOB_CATCH_MEMBER("initialization method could not be set", -1)
}


/***** zeroeth_order_statistics *****/
static auto zeroeth_order_statistics = bob::extension::VariableDoc(
  "zeroeth_order_statistics",
  "array_like <float, 1D>",
  "Returns the internal statistics. Useful to parallelize the E-step",
  ""
);
PyObject* PyBobLearnEMKMeansTrainer_getZeroethOrderStatistics(PyBobLearnEMKMeansTrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZeroethOrderStats());
  BOB_CATCH_MEMBER("zeroeth_order_statistics could not be read", 0)
}
int PyBobLearnEMKMeansTrainer_setZeroethOrderStatistics(PyBobLearnEMKMeansTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, zeroeth_order_statistics.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "zeroeth_order_statistics");
  if (!b) return -1;
  self->cxx->setZeroethOrderStats(*b);
  return 0;
  BOB_CATCH_MEMBER("zeroeth_order_statistics could not be set", -1)
}


/***** first_order_statistics *****/
static auto first_order_statistics = bob::extension::VariableDoc(
  "first_order_statistics",
  "array_like <float, 2D>",
  "Returns the internal statistics. Useful to parallelize the E-step",
  ""
);
PyObject* PyBobLearnEMKMeansTrainer_getFirstOrderStatistics(PyBobLearnEMKMeansTrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getFirstOrderStats());
  BOB_CATCH_MEMBER("first_order_statistics could not be read", 0)
}
int PyBobLearnEMKMeansTrainer_setFirstOrderStatistics(PyBobLearnEMKMeansTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, first_order_statistics.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(o, "first_order_statistics");
  if (!b) return -1;
  self->cxx->setFirstOrderStats(*b);
  return 0;
  BOB_CATCH_MEMBER("first_order_statistics could not be set", -1)
}


/***** average_min_distance *****/
static auto average_min_distance = bob::extension::VariableDoc(
  "average_min_distance",
  "str",
  "Average min (square Euclidean) distance. Useful to parallelize the E-step.",
  ""
);
PyObject* PyBobLearnEMKMeansTrainer_getAverageMinDistance(PyBobLearnEMKMeansTrainerObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("d", self->cxx->getAverageMinDistance());
  BOB_CATCH_MEMBER("Average Min Distance method could not be read", 0)
}
int PyBobLearnEMKMeansTrainer_setAverageMinDistance(PyBobLearnEMKMeansTrainerObject* self, PyObject* value, void*) {
  BOB_TRY

  if (!PyBob_NumberCheck(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an double", Py_TYPE(self)->tp_name, average_min_distance.name());
    return -1;
  }
  self->cxx->setAverageMinDistance(PyFloat_AS_DOUBLE(value));

  return 0;
  BOB_CATCH_MEMBER("Average Min Distance could not be set", -1)
}




static PyGetSetDef PyBobLearnEMKMeansTrainer_getseters[] = {
  {
   initialization_method.name(),
   (getter)PyBobLearnEMKMeansTrainer_getInitializationMethod,
   (setter)PyBobLearnEMKMeansTrainer_setInitializationMethod,
   initialization_method.doc(),
   0
  },
  {
   zeroeth_order_statistics.name(),
   (getter)PyBobLearnEMKMeansTrainer_getZeroethOrderStatistics,
   (setter)PyBobLearnEMKMeansTrainer_setZeroethOrderStatistics,
   zeroeth_order_statistics.doc(),
   0
  },
  {
   first_order_statistics.name(),
   (getter)PyBobLearnEMKMeansTrainer_getFirstOrderStatistics,
   (setter)PyBobLearnEMKMeansTrainer_setFirstOrderStatistics,
   first_order_statistics.doc(),
   0
  },
  {
   average_min_distance.name(),
   (getter)PyBobLearnEMKMeansTrainer_getAverageMinDistance,
   (setter)PyBobLearnEMKMeansTrainer_setAverageMinDistance,
   average_min_distance.doc(),
   0
  },
  {0}  // Sentinel
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

/*** initialize ***/
static auto initialize = bob::extension::FunctionDoc(
  "initialize",
  "Initialise the means randomly",
  "Data is split into as many chunks as there are means, then each mean is set to a random example within each chunk.",
  true
)
.add_prototype("kmeans_machine, data, [rng]")
.add_parameter("kmeans_machine", ":py:class:`bob.learn.em.KMeansMachine`", "KMeansMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data")
.add_parameter("rng", ":py:class:`bob.core.random.mt19937`", "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.");
static PyObject* PyBobLearnEMKMeansTrainer_initialize(PyBobLearnEMKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnEMKMeansMachineObject* kmeans_machine = 0;
  PyBlitzArrayObject* data                        = 0;
  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&|O!", kwlist, &PyBobLearnEMKMeansMachine_Type, &kmeans_machine,
                                                                 &PyBlitzArray_Converter, &data,
                                                                  &PyBoostMt19937_Type, &rng)) return 0;
  auto data_ = make_safe(data);

  // perform check on the input
  if (data->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, initialize.name());
    return 0;
  }

  if (data->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, initialize.name());
    return 0;
  }

  if (data->shape[1] != (Py_ssize_t)kmeans_machine->cxx->getNInputs() ) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [N, %" PY_FORMAT_SIZE_T "d] not [N, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, kmeans_machine->cxx->getNInputs(), data->shape[1], initialize.name());
    return 0;
  }

  if(rng){
    self->cxx->setRng(rng->rng);
  }

  self->cxx->initialize(*kmeans_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));

  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}


/*** e_step ***/
static auto e_step = bob::extension::FunctionDoc(
  "e_step",
  "Compute the E-step, which is basically the distances ",
  "Accumulate across the dataset:"
  " -zeroeth and first order statistics"
  " -average (Square Euclidean) distance from the closest mean",
  true
)
.add_prototype("kmeans_machine,data")
.add_parameter("kmeans_machine", ":py:class:`bob.learn.em.KMeansMachine`", "KMeansMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnEMKMeansTrainer_e_step(PyBobLearnEMKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = e_step.kwlist(0);

  PyBobLearnEMKMeansMachineObject* kmeans_machine;
  PyBlitzArrayObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnEMKMeansMachine_Type, &kmeans_machine,
                                                                 &PyBlitzArray_Converter, &data)) return 0;
  auto data_ = make_safe(data);

  if (data->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, e_step.name());
    return 0;
  }

  if (data->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, e_step.name());
    return 0;
  }

  if (data->shape[1] != (Py_ssize_t)kmeans_machine->cxx->getNInputs() ) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [N, %" PY_FORMAT_SIZE_T "d] not [N, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, kmeans_machine->cxx->getNInputs(), data->shape[1], e_step.name());
    return 0;
  }

  self->cxx->eStep(*kmeans_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));


  BOB_CATCH_MEMBER("cannot perform the e_step method", 0)

  Py_RETURN_NONE;
}


/*** m_step ***/
static auto m_step = bob::extension::FunctionDoc(
  "m_step",
  "Updates the mean based on the statistics from the E-step",
  0,
  true
)
.add_prototype("kmeans_machine, [data]")
.add_parameter("kmeans_machine", ":py:class:`bob.learn.em.KMeansMachine`", "KMeansMachine Object")
.add_parameter("data", "object", "Ignored.");
static PyObject* PyBobLearnEMKMeansTrainer_m_step(PyBobLearnEMKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = m_step.kwlist(0);

  PyBobLearnEMKMeansMachineObject* kmeans_machine;
  PyObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", kwlist, &PyBobLearnEMKMeansMachine_Type, &kmeans_machine,
                                                                 &data)) return 0;
  self->cxx->mStep(*kmeans_machine->cxx);

  BOB_CATCH_MEMBER("cannot perform the m_step method", 0)

  Py_RETURN_NONE;
}


/*** computeLikelihood ***/
static auto compute_likelihood = bob::extension::FunctionDoc(
  "compute_likelihood",
  "This functions returns the average min (Square Euclidean) distance (average distance to the closest mean)",
  0,
  true
)
.add_prototype("kmeans_machine")
.add_parameter("kmeans_machine", ":py:class:`bob.learn.em.KMeansMachine`", "KMeansMachine Object");
static PyObject* PyBobLearnEMKMeansTrainer_compute_likelihood(PyBobLearnEMKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = compute_likelihood.kwlist(0);

  PyBobLearnEMKMeansMachineObject* kmeans_machine;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMKMeansMachine_Type, &kmeans_machine)) return 0;

  double value = self->cxx->computeLikelihood(*kmeans_machine->cxx);
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot perform the computeLikelihood method", 0)
}


/*** reset_accumulators ***/
static auto reset_accumulators = bob::extension::FunctionDoc(
  "reset_accumulators",
  "Reset the statistics accumulators to the correct size and a value of zero.",
  0,
  true
)
.add_prototype("kmeans_machine")
.add_parameter("kmeans_machine", ":py:class:`bob.learn.em.KMeansMachine`", "KMeansMachine Object");
static PyObject* PyBobLearnEMKMeansTrainer_reset_accumulators(PyBobLearnEMKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = reset_accumulators.kwlist(0);

  PyBobLearnEMKMeansMachineObject* kmeans_machine;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMKMeansMachine_Type, &kmeans_machine)) return 0;

  self->cxx->resetAccumulators(*kmeans_machine->cxx);
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot perform the reset_accumulators method", 0)
}


static PyMethodDef PyBobLearnEMKMeansTrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnEMKMeansTrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    e_step.name(),
    (PyCFunction)PyBobLearnEMKMeansTrainer_e_step,
    METH_VARARGS|METH_KEYWORDS,
    e_step.doc()
  },
  {
    m_step.name(),
    (PyCFunction)PyBobLearnEMKMeansTrainer_m_step,
    METH_VARARGS|METH_KEYWORDS,
    m_step.doc()
  },
  {
    compute_likelihood.name(),
    (PyCFunction)PyBobLearnEMKMeansTrainer_compute_likelihood,
    METH_VARARGS|METH_KEYWORDS,
    compute_likelihood.doc()
  },
  {
    reset_accumulators.name(),
    (PyCFunction)PyBobLearnEMKMeansTrainer_reset_accumulators,
    METH_VARARGS|METH_KEYWORDS,
    reset_accumulators.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMKMeansTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMKMeansTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMKMeansTrainer_Type.tp_name = KMeansTrainer_doc.name();
  PyBobLearnEMKMeansTrainer_Type.tp_basicsize = sizeof(PyBobLearnEMKMeansTrainerObject);
  PyBobLearnEMKMeansTrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance
  PyBobLearnEMKMeansTrainer_Type.tp_doc = KMeansTrainer_doc.doc();

  // set the functions
  PyBobLearnEMKMeansTrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnEMKMeansTrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnEMKMeansTrainer_init);
  PyBobLearnEMKMeansTrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnEMKMeansTrainer_delete);
  PyBobLearnEMKMeansTrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMKMeansTrainer_RichCompare);
  PyBobLearnEMKMeansTrainer_Type.tp_methods = PyBobLearnEMKMeansTrainer_methods;
  PyBobLearnEMKMeansTrainer_Type.tp_getset = PyBobLearnEMKMeansTrainer_getseters;
  PyBobLearnEMKMeansTrainer_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMKMeansTrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMKMeansTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMKMeansTrainer_Type);
  return PyModule_AddObject(module, "KMeansTrainer", (PyObject*)&PyBobLearnEMKMeansTrainer_Type) >= 0;
}

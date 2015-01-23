/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Web 21 Jan 12:30:00 2015
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"
#include <boost/make_shared.hpp>

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */

static auto GMMBaseTrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GMMBaseTrainer",
  "This class implements the E-step of the expectation-maximisation"
  "algorithm for a :py:class:`bob.learn.misc.GMMMachine`"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a GMMBaseTrainer",
    "",
    true
  )
  .add_prototype("update_means, [update_variances], [update_weights], [mean_var_update_responsibilities_threshold]","")
  .add_prototype("other","")
  .add_prototype("","")

  .add_parameter("update_means", "bool", "Update means on each iteration")
  .add_parameter("update_variances", "bool", "Update variances on each iteration")
  .add_parameter("update_weights", "bool", "Update weights on each iteration")
  .add_parameter("mean_var_update_responsibilities_threshold", "float", "Threshold over the responsibilities of the Gaussians Equations 9.24, 9.25 of Bishop, `Pattern recognition and machine learning`, 2006 require a division by the responsibilities, which might be equal to zero because of numerical issue. This threshold is used to avoid such divisions.")
  .add_parameter("other", ":py:class:`bob.learn.misc.GMMBaseTrainer`", "A GMMBaseTrainer object to be copied.")
);



static int PyBobLearnMiscGMMBaseTrainer_init_copy(PyBobLearnMiscGMMBaseTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMBaseTrainer_doc.kwlist(1);
  PyBobLearnMiscGMMBaseTrainerObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMBaseTrainer_Type, &tt)){
    GMMBaseTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::GMMBaseTrainer(*tt->cxx));
  return 0;
}


static int PyBobLearnMiscGMMBaseTrainer_init_bool(PyBobLearnMiscGMMBaseTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMBaseTrainer_doc.kwlist(0);
  PyObject* update_means     = 0;
  PyObject* update_variances = 0;
  PyObject* update_weights   = 0;
  double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon();

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!O!d", kwlist, &PyBool_Type, &update_means, &PyBool_Type, 
                                                             &update_variances, &PyBool_Type, &update_weights, &mean_var_update_responsibilities_threshold)){
    GMMBaseTrainer_doc.print_usage();
    return -1;
  }
  self->cxx.reset(new bob::learn::misc::GMMBaseTrainer(f(update_means), f(update_variances), f(update_weights), mean_var_update_responsibilities_threshold));
  return 0;
}


static int PyBobLearnMiscGMMBaseTrainer_init(PyBobLearnMiscGMMBaseTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (nargs==0){ //default initializer ()
    self->cxx.reset(new bob::learn::misc::GMMBaseTrainer());
    return 0;
  }
  else{
    //Reading the input argument
    PyObject* arg = 0;
    if (PyTuple_Size(args))
      arg = PyTuple_GET_ITEM(args, 0);
    else {
      PyObject* tmp = PyDict_Values(kwargs);
      auto tmp_ = make_safe(tmp);
      arg = PyList_GET_ITEM(tmp, 0);
    }

    // If the constructor input is GMMBaseTrainer object
    if (PyBobLearnMiscGMMBaseTrainer_Check(arg))
      return PyBobLearnMiscGMMBaseTrainer_init_copy(self, args, kwargs);
    else
      return PyBobLearnMiscGMMBaseTrainer_init_bool(self, args, kwargs);
  }

  BOB_CATCH_MEMBER("cannot create GMMBaseTrainer_init_bool", 0)
  return 0;
}


static void PyBobLearnMiscGMMBaseTrainer_delete(PyBobLearnMiscGMMBaseTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnMiscGMMBaseTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscGMMBaseTrainer_Type));
}


static PyObject* PyBobLearnMiscGMMBaseTrainer_RichCompare(PyBobLearnMiscGMMBaseTrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscGMMBaseTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscGMMBaseTrainerObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare GMMBaseTrainer objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/


/***** gmm_stats *****/
static auto gmm_stats = bob::extension::VariableDoc(
  "gmm_stats",
  ":py:class:`bob.learn.misc.GMMStats`",
  "Get/Set GMMStats",
  ""
);
PyObject* PyBobLearnMiscGMMBaseTrainer_getGMMStats(PyBobLearnMiscGMMBaseTrainerObject* self, void*){
  BOB_TRY

  bob::learn::misc::GMMStats stats = self->cxx->getGMMStats();
  boost::shared_ptr<bob::learn::misc::GMMStats> stats_shared = boost::make_shared<bob::learn::misc::GMMStats>(stats);

  //Allocating the correspondent python object
  PyBobLearnMiscGMMStatsObject* retval =
    (PyBobLearnMiscGMMStatsObject*)PyBobLearnMiscGMMStats_Type.tp_alloc(&PyBobLearnMiscGMMStats_Type, 0);

  retval->cxx = stats_shared;

  return Py_BuildValue("O",retval);
  BOB_CATCH_MEMBER("GMMStats could not be read", 0)
}
/*
int PyBobLearnMiscGMMBaseTrainer_setGMMStats(PyBobLearnMiscGMMBaseTrainerObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBobLearnMiscGMMStats_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a :py:class:`bob.learn.misc.GMMStats`", Py_TYPE(self)->tp_name, gmm_stats.name());
    return -1;
  }

  PyBobLearnMiscGMMStatsObject* stats = 0;
  PyArg_Parse(value, "O!", &PyBobLearnMiscGMMStats_Type,&stats);

  self->cxx->setGMMStats(*stats->cxx);

  return 0;
  BOB_CATCH_MEMBER("gmm_stats could not be set", -1)  
}
*/


/***** update_means *****/
static auto update_means = bob::extension::VariableDoc(
  "update_means",
  "bool",
  "Update means on each iteration",
  ""
);
PyObject* PyBobLearnMiscGMMBaseTrainer_getUpdateMeans(PyBobLearnMiscGMMBaseTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("O",self->cxx->getUpdateMeans()?Py_True:Py_False);
  BOB_CATCH_MEMBER("update_means could not be read", 0)
}

/***** update_variances *****/
static auto update_variances = bob::extension::VariableDoc(
  "update_variances",
  "bool",
  "Update variances on each iteration",
  ""
);
PyObject* PyBobLearnMiscGMMBaseTrainer_getUpdateVariances(PyBobLearnMiscGMMBaseTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("O",self->cxx->getUpdateVariances()?Py_True:Py_False);
  BOB_CATCH_MEMBER("update_variances could not be read", 0)
}


/***** update_weights *****/
static auto update_weights = bob::extension::VariableDoc(
  "update_weights",
  "bool",
  "Update weights on each iteration",
  ""
);
PyObject* PyBobLearnMiscGMMBaseTrainer_getUpdateWeights(PyBobLearnMiscGMMBaseTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("O",self->cxx->getUpdateWeights()?Py_True:Py_False);
  BOB_CATCH_MEMBER("update_weights could not be read", 0)
}


    
     

/***** mean_var_update_responsibilities_threshold *****/
static auto mean_var_update_responsibilities_threshold = bob::extension::VariableDoc(
  "mean_var_update_responsibilities_threshold",
  "bool",
  "Threshold over the responsibilities of the Gaussians" 
  "Equations 9.24, 9.25 of Bishop, \"Pattern recognition and machine learning\", 2006" 
  "require a division by the responsibilities, which might be equal to zero" 
  "because of numerical issue. This threshold is used to avoid such divisions.",
  ""
);
PyObject* PyBobLearnMiscGMMBaseTrainer_getMeanVarUpdateResponsibilitiesThreshold(PyBobLearnMiscGMMBaseTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getMeanVarUpdateResponsibilitiesThreshold());
  BOB_CATCH_MEMBER("update_weights could not be read", 0)
}


static PyGetSetDef PyBobLearnMiscGMMBaseTrainer_getseters[] = { 
  {
    update_means.name(),
    (getter)PyBobLearnMiscGMMBaseTrainer_getUpdateMeans,
    0,
    update_means.doc(),
    0
  },
  {
    update_variances.name(),
    (getter)PyBobLearnMiscGMMBaseTrainer_getUpdateVariances,
    0,
    update_variances.doc(),
    0
  },
  {
    update_weights.name(),
    (getter)PyBobLearnMiscGMMBaseTrainer_getUpdateWeights,
    0,
    update_weights.doc(),
    0
  },  
  {
    mean_var_update_responsibilities_threshold.name(),
    (getter)PyBobLearnMiscGMMBaseTrainer_getMeanVarUpdateResponsibilitiesThreshold,
    0,
    mean_var_update_responsibilities_threshold.doc(),
    0
  },  
  {
    gmm_stats.name(),
    (getter)PyBobLearnMiscGMMBaseTrainer_getGMMStats,
    0, //(setter)PyBobLearnMiscGMMBaseTrainer_setGMMStats,
    gmm_stats.doc(),
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
  "Initialization before the EM steps",
  "Instanciate :py:class:`bob.learn.misc.GMMStats`",
  true
)
.add_prototype("gmm_machine")
.add_parameter("gmm_machine", ":py:class:`bob.learn.misc.GMMMachine`", "GMMMachine Object");
static PyObject* PyBobLearnMiscGMMBaseTrainer_initialize(PyBobLearnMiscGMMBaseTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnMiscGMMMachineObject* gmm_machine = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMMachine_Type, &gmm_machine)) Py_RETURN_NONE;

  self->cxx->initialize(*gmm_machine->cxx);

  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}



/*** eStep ***/
static auto eStep = bob::extension::FunctionDoc(
  "eStep",
  "Calculates and saves statistics across the dataset,"
  "and saves these as m_ss. ",

  "Calculates the average log likelihood of the observations given the GMM,"
  "and returns this in average_log_likelihood."
  "The statistics, m_ss, will be used in the mStep() that follows.",

  true
)
.add_prototype("gmm_machine,data")
.add_parameter("gmm_machine", ":py:class:`bob.learn.misc.GMMMachine`", "GMMMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnMiscGMMBaseTrainer_eStep(PyBobLearnMiscGMMBaseTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = eStep.kwlist(0);

  PyBobLearnMiscGMMMachineObject* gmm_machine;
  PyBlitzArrayObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnMiscGMMMachine_Type, &gmm_machine,
                                                                 &PyBlitzArray_Converter, &data)) Py_RETURN_NONE;
  auto data_ = make_safe(data);

  self->cxx->eStep(*gmm_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));

  BOB_CATCH_MEMBER("cannot perform the eStep method", 0)

  Py_RETURN_NONE;
}


/*** computeLikelihood ***/
static auto compute_likelihood = bob::extension::FunctionDoc(
  "compute_likelihood",
  "This functions returns the average min (Square Euclidean) distance (average distance to the closest mean)",
  0,
  true
)
.add_prototype("gmm_machine")
.add_parameter("gmm_machine", ":py:class:`bob.learn.misc.GMMMachine`", "GMMMachine Object");
static PyObject* PyBobLearnMiscGMMBaseTrainer_compute_likelihood(PyBobLearnMiscGMMBaseTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = compute_likelihood.kwlist(0);

  PyBobLearnMiscGMMMachineObject* gmm_machine;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMMachine_Type, &gmm_machine)) Py_RETURN_NONE;

  double value = self->cxx->computeLikelihood(*gmm_machine->cxx);
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot perform the computeLikelihood method", 0)
}


static PyMethodDef PyBobLearnMiscGMMBaseTrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnMiscGMMBaseTrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    eStep.name(),
    (PyCFunction)PyBobLearnMiscGMMBaseTrainer_eStep,
    METH_VARARGS|METH_KEYWORDS,
    eStep.doc()
  },
  {
    compute_likelihood.name(),
    (PyCFunction)PyBobLearnMiscGMMBaseTrainer_compute_likelihood,
    METH_VARARGS|METH_KEYWORDS,
    compute_likelihood.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscGMMBaseTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscGMMBaseTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscGMMBaseTrainer_Type.tp_name      = GMMBaseTrainer_doc.name();
  PyBobLearnMiscGMMBaseTrainer_Type.tp_basicsize = sizeof(PyBobLearnMiscGMMBaseTrainerObject);
  PyBobLearnMiscGMMBaseTrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscGMMBaseTrainer_Type.tp_doc       = GMMBaseTrainer_doc.doc();

  // set the functions
  PyBobLearnMiscGMMBaseTrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnMiscGMMBaseTrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnMiscGMMBaseTrainer_init);
  PyBobLearnMiscGMMBaseTrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnMiscGMMBaseTrainer_delete);
  PyBobLearnMiscGMMBaseTrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscGMMBaseTrainer_RichCompare);
  PyBobLearnMiscGMMBaseTrainer_Type.tp_methods      = PyBobLearnMiscGMMBaseTrainer_methods;
  PyBobLearnMiscGMMBaseTrainer_Type.tp_getset       = PyBobLearnMiscGMMBaseTrainer_getseters;
  PyBobLearnMiscGMMBaseTrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnMiscGMMBaseTrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscGMMBaseTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscGMMBaseTrainer_Type);
  return PyModule_AddObject(module, "GMMBaseTrainer", (PyObject*)&PyBobLearnMiscGMMBaseTrainer_Type) >= 0;
}


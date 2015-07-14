/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Web 22 Jan 16:45:00 2015
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */

static auto ML_GMMTrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".ML_GMMTrainer",
  "This class implements the maximum likelihood M-step of the expectation-maximisation algorithm for a GMM Machine."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a ML_GMMTrainer",
    "",
    true
  )
  .add_prototype("update_means, [update_variances], [update_weights], [mean_var_update_responsibilities_threshold]","")
  .add_prototype("other","")

  .add_parameter("update_means", "bool", "Update means on each iteration")
  .add_parameter("update_variances", "bool", "Update variances on each iteration")
  .add_parameter("update_weights", "bool", "Update weights on each iteration")
  .add_parameter("mean_var_update_responsibilities_threshold", "float", "Threshold over the responsibilities of the Gaussians Equations 9.24, 9.25 of Bishop, `Pattern recognition and machine learning`, 2006 require a division by the responsibilities, which might be equal to zero because of numerical issue. This threshold is used to avoid such divisions.")


  .add_parameter("other", ":py:class:`bob.learn.em.ML_GMMTrainer`", "A ML_GMMTrainer object to be copied.")
);


static int PyBobLearnEMMLGMMTrainer_init_copy(PyBobLearnEMMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ML_GMMTrainer_doc.kwlist(1);
  PyBobLearnEMMLGMMTrainerObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMMLGMMTrainer_Type, &o)){
    ML_GMMTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::ML_GMMTrainer(*o->cxx));
  return 0;
}


static int PyBobLearnEMMLGMMTrainer_init_base_trainer(PyBobLearnEMMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ML_GMMTrainer_doc.kwlist(0);

  PyObject* update_means     = Py_True;
  PyObject* update_variances = Py_False;
  PyObject* update_weights   = Py_False;
  double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon();

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O!O!O!d", kwlist,
                                   &PyBool_Type, &update_means,
                                   &PyBool_Type, &update_variances,
                                   &PyBool_Type, &update_weights,
                                   &mean_var_update_responsibilities_threshold)){
    ML_GMMTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::ML_GMMTrainer(f(update_means), f(update_variances), f(update_weights),
                                                       mean_var_update_responsibilities_threshold));
  return 0;
}



static int PyBobLearnEMMLGMMTrainer_init(PyBobLearnEMMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (nargs==0)
    return PyBobLearnEMMLGMMTrainer_init_base_trainer(self, args, kwargs);
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
    if (PyBobLearnEMMLGMMTrainer_Check(arg))
      return PyBobLearnEMMLGMMTrainer_init_copy(self, args, kwargs);
    else
      return PyBobLearnEMMLGMMTrainer_init_base_trainer(self, args, kwargs);
  }
  BOB_CATCH_MEMBER("cannot create GMMBaseTrainer_init_bool", -1)
  return 0;
}


static void PyBobLearnEMMLGMMTrainer_delete(PyBobLearnEMMLGMMTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnEMMLGMMTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMMLGMMTrainer_Type));
}


static PyObject* PyBobLearnEMMLGMMTrainer_RichCompare(PyBobLearnEMMLGMMTrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMMLGMMTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMMLGMMTrainerObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare ML_GMMTrainer objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto gmm_statistics = bob::extension::VariableDoc(
  "gmm_statistics",
  ":py:class:`GMMStats`",
  "The GMM statistics that were used internally in the E- and M-steps",
  "Setting and getting the internal GMM statistics might be useful to parallelize the GMM training."
);
PyObject* PyBobLearnEMMLGMMTrainer_get_gmm_statistics(PyBobLearnEMMLGMMTrainerObject* self, void*){
  BOB_TRY
  PyBobLearnEMGMMStatsObject* stats = (PyBobLearnEMGMMStatsObject*)PyBobLearnEMGMMStats_Type.tp_alloc(&PyBobLearnEMGMMStats_Type, 0);
  stats->cxx = self->cxx->base_trainer().getGMMStats();
  return Py_BuildValue("N", stats);
  BOB_CATCH_MEMBER("gmm_statistics could not be read", 0)
}
int PyBobLearnEMMLGMMTrainer_set_gmm_statistics(PyBobLearnEMMLGMMTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyBobLearnEMGMMStats_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a GMMStats object", Py_TYPE(self)->tp_name, gmm_statistics.name());
    return -1;
  }
  self->cxx->base_trainer().setGMMStats(reinterpret_cast<PyBobLearnEMGMMStatsObject*>(value)->cxx);
  return 0;
  BOB_CATCH_MEMBER("gmm_statistics could not be set", -1)
}


static PyGetSetDef PyBobLearnEMMLGMMTrainer_getseters[] = {
  {
   gmm_statistics.name(),
   (getter)PyBobLearnEMMLGMMTrainer_get_gmm_statistics,
   (setter)PyBobLearnEMMLGMMTrainer_set_gmm_statistics,
   gmm_statistics.doc(),
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
  "",
  true
)
.add_prototype("gmm_machine, [data]")
.add_parameter("gmm_machine", ":py:class:`bob.learn.em.GMMMachine`", "GMMMachine Object")
.add_parameter("data", "object", "Ignored.");
static PyObject* PyBobLearnEMMLGMMTrainer_initialize(PyBobLearnEMMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);
  PyBobLearnEMGMMMachineObject* gmm_machine = 0;
  PyObject* data;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", kwlist, &PyBobLearnEMGMMMachine_Type, &gmm_machine,
                                                                  &data)) return 0;

  self->cxx->initialize(*gmm_machine->cxx);
  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}


/*** e_step ***/
static auto e_step = bob::extension::FunctionDoc(
  "e_step",
  "Calculates and saves statistics across the dataset,"
  "and saves these as m_ss. ",

  "Calculates the average log likelihood of the observations given the GMM,"
  "and returns this in average_log_likelihood."
  "The statistics, :py:attr:`gmm_statistics`, will be used in the :py:func:`m_step` that follows.",

  true
)
.add_prototype("gmm_machine,data")
.add_parameter("gmm_machine", ":py:class:`bob.learn.em.GMMMachine`", "GMMMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnEMMLGMMTrainer_e_step(PyBobLearnEMMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = e_step.kwlist(0);

  PyBobLearnEMGMMMachineObject* gmm_machine;
  PyBlitzArrayObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnEMGMMMachine_Type, &gmm_machine,
                                                                 &PyBlitzArray_Converter, &data)) return 0;
  auto data_ = make_safe(data);

  // perform check on the input
  if (data->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, e_step.name());
    return 0;
  }

  if (data->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, e_step.name());
    return 0;
  }

  if (data->shape[1] != (Py_ssize_t)gmm_machine->cxx->getNInputs() ) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [N, %" PY_FORMAT_SIZE_T "d] not [N, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, gmm_machine->cxx->getNInputs(), data->shape[1], e_step.name());
    return 0;
  }

  self->cxx->eStep(*gmm_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));

  BOB_CATCH_MEMBER("cannot perform the e_step method", 0)

  Py_RETURN_NONE;
}


/*** m_step ***/
static auto m_step = bob::extension::FunctionDoc(
  "m_step",
  "Performs a maximum likelihood (ML) update of the GMM parameters "
  "using the accumulated statistics in :py:attr:`gmm_statistics`",

  "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",

  true
)
.add_prototype("gmm_machine, [data]")
.add_parameter("gmm_machine", ":py:class:`bob.learn.em.GMMMachine`", "GMMMachine Object")
.add_parameter("data", "object", "Ignored.");
static PyObject* PyBobLearnEMMLGMMTrainer_m_step(PyBobLearnEMMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = m_step.kwlist(0);

  PyBobLearnEMGMMMachineObject* gmm_machine = 0;
  PyObject* data;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", kwlist, &PyBobLearnEMGMMMachine_Type, &gmm_machine,
                                                                 &data)) return 0;

  self->cxx->mStep(*gmm_machine->cxx);

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
.add_prototype("gmm_machine")
.add_parameter("gmm_machine", ":py:class:`bob.learn.em.GMMMachine`", "GMMMachine Object");
static PyObject* PyBobLearnEMMLGMMTrainer_compute_likelihood(PyBobLearnEMMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = compute_likelihood.kwlist(0);

  PyBobLearnEMGMMMachineObject* gmm_machine;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMGMMMachine_Type, &gmm_machine)) return 0;

  double value = self->cxx->computeLikelihood(*gmm_machine->cxx);
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot perform the computeLikelihood method", 0)
}



static PyMethodDef PyBobLearnEMMLGMMTrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnEMMLGMMTrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    e_step.name(),
    (PyCFunction)PyBobLearnEMMLGMMTrainer_e_step,
    METH_VARARGS|METH_KEYWORDS,
    e_step.doc()
  },
  {
    m_step.name(),
    (PyCFunction)PyBobLearnEMMLGMMTrainer_m_step,
    METH_VARARGS|METH_KEYWORDS,
    m_step.doc()
  },
  {
    compute_likelihood.name(),
    (PyCFunction)PyBobLearnEMMLGMMTrainer_compute_likelihood,
    METH_VARARGS|METH_KEYWORDS,
    compute_likelihood.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMMLGMMTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMMLGMMTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMMLGMMTrainer_Type.tp_name      = ML_GMMTrainer_doc.name();
  PyBobLearnEMMLGMMTrainer_Type.tp_basicsize = sizeof(PyBobLearnEMMLGMMTrainerObject);
  PyBobLearnEMMLGMMTrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance
  PyBobLearnEMMLGMMTrainer_Type.tp_doc       = ML_GMMTrainer_doc.doc();

  // set the functions
  PyBobLearnEMMLGMMTrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnEMMLGMMTrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnEMMLGMMTrainer_init);
  PyBobLearnEMMLGMMTrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnEMMLGMMTrainer_delete);
  PyBobLearnEMMLGMMTrainer_Type.tp_richcompare  = reinterpret_cast<richcmpfunc>(PyBobLearnEMMLGMMTrainer_RichCompare);
  PyBobLearnEMMLGMMTrainer_Type.tp_methods      = PyBobLearnEMMLGMMTrainer_methods;
  PyBobLearnEMMLGMMTrainer_Type.tp_getset       = PyBobLearnEMMLGMMTrainer_getseters;
  PyBobLearnEMMLGMMTrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnEMMLGMMTrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMMLGMMTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMMLGMMTrainer_Type);
  return PyModule_AddObject(module, "ML_GMMTrainer", (PyObject*)&PyBobLearnEMMLGMMTrainer_Type) >= 0;
}

/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Web 23 Jan 16:42:00 2015
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

static auto MAP_GMMTrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".MAP_GMMTrainer",
  "This class implements the maximum a posteriori M-step of the expectation-maximization algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a MAP_GMMTrainer",
    "Additionally to the copy constructor, there are two different ways to call this constructor, one using the ``relevance_factor`` and one using the ``alpha``, both which have the same signature. "
    "Hence, the only way to differentiate the two functions is by using keyword arguments.",
    true
  )

  .add_prototype("prior_gmm, relevance_factor, [update_means], [update_variances], [update_weights], [mean_var_update_responsibilities_threshold]","")
  .add_prototype("prior_gmm, alpha, [update_means], [update_variances], [update_weights], [mean_var_update_responsibilities_threshold]","")
  .add_prototype("other","")

  .add_parameter("prior_gmm", ":py:class:`bob.learn.em.GMMMachine`", "The prior GMM to be adapted (Universal Background Model UBM).")
  .add_parameter("relevance_factor", "float", "If set the Reynolds Adaptation procedure will be  applied. See Eq (14) from [Reynolds2000]_")
  .add_parameter("alpha", "float", "Set directly the alpha parameter (Eq (14) from [Reynolds2000]_), ignoring zeroth order statistics as a weighting factor.")

  .add_parameter("update_means", "bool", "[Default: ``True``] Update means on each iteration")
  .add_parameter("update_variances", "bool", "[Default: ``True``] Update variances on each iteration")
  .add_parameter("update_weights", "bool", "[Default: ``True``] Update weights on each iteration")
  .add_parameter("mean_var_update_responsibilities_threshold", "float", "[Default: min_float] Threshold over the responsibilities of the Gaussians Equations 9.24, 9.25 of Bishop, `Pattern recognition and machine learning`, 2006 require a division by the responsibilities, which might be equal to zero because of numerical issue. This threshold is used to avoid such divisions.")

  .add_parameter("other", ":py:class:`bob.learn.em.MAP_GMMTrainer`", "A MAP_GMMTrainer object to be copied.")
);


static int PyBobLearnEMMAPGMMTrainer_init_copy(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = MAP_GMMTrainer_doc.kwlist(2);
  PyBobLearnEMMAPGMMTrainerObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMMAPGMMTrainer_Type, &o)){
    MAP_GMMTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::MAP_GMMTrainer(*o->cxx));
  return 0;
}


static int PyBobLearnEMMAPGMMTrainer_init_base_trainer(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist1 = MAP_GMMTrainer_doc.kwlist(0);
  char** kwlist2 = MAP_GMMTrainer_doc.kwlist(1);

  PyBobLearnEMGMMMachineObject* gmm_machine;
  bool reynolds_adaptation   = false;
  double alpha = 0.5;
  double relevance_factor = 4.0;
  double aux = 0;

  PyObject* update_means     = Py_True;
  PyObject* update_variances = Py_False;
  PyObject* update_weights   = Py_False;
  double mean_var_update_responsibilities_threshold = std::numeric_limits<double>::epsilon();

  PyObject* keyword_relevance_factor = Py_BuildValue("s", kwlist1[1]);
  PyObject* keyword_alpha            = Py_BuildValue("s", kwlist2[1]);

  auto keyword_relevance_factor_ = make_safe(keyword_relevance_factor);
  auto keyword_alpha_            = make_safe(keyword_alpha);

  //Here we have to select which keyword argument to read
  if (kwargs && PyDict_Contains(kwargs, keyword_relevance_factor)){
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!d|O!O!O!d", kwlist1,
          &PyBobLearnEMGMMMachine_Type, &gmm_machine,
          &aux,
          &PyBool_Type, &update_means,
          &PyBool_Type, &update_variances,
          &PyBool_Type, &update_weights,
          &mean_var_update_responsibilities_threshold))
      return -1;
    reynolds_adaptation = true;
  } else if (kwargs && PyDict_Contains(kwargs, keyword_alpha)){
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!d|O!O!O!d", kwlist2,
          &PyBobLearnEMGMMMachine_Type, &gmm_machine,
          &aux,
          &PyBool_Type, &update_means,
          &PyBool_Type, &update_variances,
          &PyBool_Type, &update_weights,
          &mean_var_update_responsibilities_threshold))
      return -1;
    reynolds_adaptation = false;
  } else {
    PyErr_Format(PyExc_RuntimeError, "%s. One of the two keyword arguments '%s' or '%s' must be present.", Py_TYPE(self)->tp_name, kwlist1[1], kwlist2[1]);
    MAP_GMMTrainer_doc.print_usage();
    return -1;
  }

  if (reynolds_adaptation)
    relevance_factor = aux;
  else
    alpha = aux;


  self->cxx.reset(new bob::learn::em::MAP_GMMTrainer(f(update_means), f(update_variances), f(update_weights),
                                                       mean_var_update_responsibilities_threshold,
                                                       reynolds_adaptation, relevance_factor, alpha, gmm_machine->cxx));
  return 0;

}



static int PyBobLearnEMMAPGMMTrainer_init(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (nargs==0){
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s (see help)", Py_TYPE(self)->tp_name);
    MAP_GMMTrainer_doc.print_usage();
    return -1;
  }

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
  if(PyBobLearnEMMAPGMMTrainer_Check(arg))
    return PyBobLearnEMMAPGMMTrainer_init_copy(self, args, kwargs);
  else{
    return PyBobLearnEMMAPGMMTrainer_init_base_trainer(self, args, kwargs);
  }

  BOB_CATCH_MEMBER("cannot create MAP_GMMTrainer", -1)
  return 0;
}


static void PyBobLearnEMMAPGMMTrainer_delete(PyBobLearnEMMAPGMMTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnEMMAPGMMTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMMAPGMMTrainer_Type));
}


static PyObject* PyBobLearnEMMAPGMMTrainer_RichCompare(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMMAPGMMTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMMAPGMMTrainerObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare MAP_GMMTrainer objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** relevance_factor *****/
static auto relevance_factor = bob::extension::VariableDoc(
  "relevance_factor",
  "float",
  "If set the reynolds_adaptation parameters, will apply the Reynolds Adaptation Factor. See Eq (14) from [Reynolds2000]_",
  ""
);
PyObject* PyBobLearnEMMAPGMMTrainer_getRelevanceFactor(PyBobLearnEMMAPGMMTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getRelevanceFactor());
  BOB_CATCH_MEMBER("relevance_factor could not be read", 0)
}
int PyBobLearnEMMAPGMMTrainer_setRelevanceFactor(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* value, void*){
  BOB_TRY

  if(!PyBob_NumberCheck(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a float", Py_TYPE(self)->tp_name, relevance_factor.name());
    return -1;
  }

  self->cxx->setRelevanceFactor(PyFloat_AS_DOUBLE(value));
  return 0;
  BOB_CATCH_MEMBER("relevance_factor could not be set", -1)
}


/***** alpha *****/
static auto alpha = bob::extension::VariableDoc(
  "alpha",
  "float",
  "Set directly the alpha parameter (Eq (14) from [Reynolds2000]_), ignoring zeroth order statistics as a weighting factor.",
  ""
);
PyObject* PyBobLearnEMMAPGMMTrainer_getAlpha(PyBobLearnEMMAPGMMTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getAlpha());
  BOB_CATCH_MEMBER("alpha could not be read", 0)
}
int PyBobLearnEMMAPGMMTrainer_setAlpha(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* value, void*){
  BOB_TRY

  if(!PyBob_NumberCheck(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a float", Py_TYPE(self)->tp_name, alpha.name());
    return -1;
  }

  self->cxx->setAlpha(PyFloat_AS_DOUBLE(value));
  return 0;
  BOB_CATCH_MEMBER("alpha could not be set", -1)
}


static auto gmm_statistics = bob::extension::VariableDoc(
  "gmm_statistics",
  ":py:class:`GMMStats`",
  "The GMM statistics that were used internally in the E- and M-steps",
  "Setting and getting the internal GMM statistics might be useful to parallelize the GMM training."
);
PyObject* PyBobLearnEMMAPGMMTrainer_get_gmm_statistics(PyBobLearnEMMAPGMMTrainerObject* self, void*){
  BOB_TRY
  PyBobLearnEMGMMStatsObject* stats = (PyBobLearnEMGMMStatsObject*)PyBobLearnEMGMMStats_Type.tp_alloc(&PyBobLearnEMGMMStats_Type, 0);
  stats->cxx = self->cxx->base_trainer().getGMMStats();
  return Py_BuildValue("N", stats);
  BOB_CATCH_MEMBER("gmm_statistics could not be read", 0)
}
int PyBobLearnEMMAPGMMTrainer_set_gmm_statistics(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyBobLearnEMGMMStats_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a GMMStats object", Py_TYPE(self)->tp_name, gmm_statistics.name());
    return -1;
  }
  self->cxx->base_trainer().setGMMStats(reinterpret_cast<PyBobLearnEMGMMStatsObject*>(value)->cxx);
  return 0;
  BOB_CATCH_MEMBER("gmm_statistics could not be set", -1)
}


static PyGetSetDef PyBobLearnEMMAPGMMTrainer_getseters[] = {
  {
    alpha.name(),
    (getter)PyBobLearnEMMAPGMMTrainer_getAlpha,
    (setter)PyBobLearnEMMAPGMMTrainer_setAlpha,
    alpha.doc(),
    0
  },
  {
    relevance_factor.name(),
    (getter)PyBobLearnEMMAPGMMTrainer_getRelevanceFactor,
    (setter)PyBobLearnEMMAPGMMTrainer_setRelevanceFactor,
    relevance_factor.doc(),
    0
  },
  {
    gmm_statistics.name(),
    (getter)PyBobLearnEMMAPGMMTrainer_get_gmm_statistics,
    (setter)PyBobLearnEMMAPGMMTrainer_set_gmm_statistics,
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
static PyObject* PyBobLearnEMMAPGMMTrainer_initialize(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
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
  "Calculates and saves statistics across the dataset and saves these as :py:attr:`gmm_statistics`. ",

  "Calculates the average log likelihood of the observations given the GMM,"
  "and returns this in average_log_likelihood."
  "The statistics, :py:attr:`gmm_statistics`, will be used in the :py:meth:`m_step` that follows.",

  true
)
.add_prototype("gmm_machine,data")
.add_parameter("gmm_machine", ":py:class:`bob.learn.em.GMMMachine`", "GMMMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnEMMAPGMMTrainer_e_step(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
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

   "Performs a maximum a posteriori (MAP) update of the GMM parameters using the accumulated statistics in :py:attr:`gmm_statistics` and the parameters of the prior model",
  "",
  true
)
.add_prototype("gmm_machine, [data]")
.add_parameter("gmm_machine", ":py:class:`bob.learn.em.GMMMachine`", "GMMMachine Object")
.add_parameter("data", "object", "Ignored.");
static PyObject* PyBobLearnEMMAPGMMTrainer_m_step(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = m_step.kwlist(0);

  PyBobLearnEMGMMMachineObject* gmm_machine;
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
static PyObject* PyBobLearnEMMAPGMMTrainer_compute_likelihood(PyBobLearnEMMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = compute_likelihood.kwlist(0);

  PyBobLearnEMGMMMachineObject* gmm_machine;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMGMMMachine_Type, &gmm_machine)) return 0;

  double value = self->cxx->computeLikelihood(*gmm_machine->cxx);
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot perform the computeLikelihood method", 0)
}



static PyMethodDef PyBobLearnEMMAPGMMTrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnEMMAPGMMTrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    e_step.name(),
    (PyCFunction)PyBobLearnEMMAPGMMTrainer_e_step,
    METH_VARARGS|METH_KEYWORDS,
    e_step.doc()
  },
  {
    m_step.name(),
    (PyCFunction)PyBobLearnEMMAPGMMTrainer_m_step,
    METH_VARARGS|METH_KEYWORDS,
    m_step.doc()
  },
  {
    compute_likelihood.name(),
    (PyCFunction)PyBobLearnEMMAPGMMTrainer_compute_likelihood,
    METH_VARARGS|METH_KEYWORDS,
    compute_likelihood.doc()
  },

  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMMAPGMMTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMMAPGMMTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMMAPGMMTrainer_Type.tp_name      = MAP_GMMTrainer_doc.name();
  PyBobLearnEMMAPGMMTrainer_Type.tp_basicsize = sizeof(PyBobLearnEMMAPGMMTrainerObject);
  PyBobLearnEMMAPGMMTrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance
  PyBobLearnEMMAPGMMTrainer_Type.tp_doc       = MAP_GMMTrainer_doc.doc();

  // set the functions
  PyBobLearnEMMAPGMMTrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnEMMAPGMMTrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnEMMAPGMMTrainer_init);
  PyBobLearnEMMAPGMMTrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnEMMAPGMMTrainer_delete);
  PyBobLearnEMMAPGMMTrainer_Type.tp_richcompare  = reinterpret_cast<richcmpfunc>(PyBobLearnEMMAPGMMTrainer_RichCompare);
  PyBobLearnEMMAPGMMTrainer_Type.tp_methods      = PyBobLearnEMMAPGMMTrainer_methods;
  PyBobLearnEMMAPGMMTrainer_Type.tp_getset       = PyBobLearnEMMAPGMMTrainer_getseters;
  PyBobLearnEMMAPGMMTrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnEMMAPGMMTrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMMAPGMMTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMMAPGMMTrainer_Type);
  return PyModule_AddObject(module, "MAP_GMMTrainer", (PyObject*)&PyBobLearnEMMAPGMMTrainer_Type) >= 0;
}

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
  "This class implements the maximum a posteriori M-step of the expectation-maximisation algorithm for a GMM Machine. The prior parameters are encoded in the form of a GMM (e.g. a universal background model). The EM algorithm thus performs GMM adaptation."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a MAP_GMMTrainer",
    "",
    true
  )
  
  
  //.add_prototype("gmm_base_trainer,prior_gmm,[reynolds_adaptation],[relevance_factor],[alpha]","")
  .add_prototype("gmm_base_trainer, prior_gmm, reynolds_adaptation, [relevance_factor], [alpha]","")
  .add_prototype("other","")
  .add_prototype("","")

  .add_parameter("gmm_base_trainer", ":py:class:`bob.learn.misc.GMMBaseTrainer`", "A GMMBaseTrainer object.")
  .add_parameter("prior_gmm", ":py:class:`bob.learn.misc.GMMMachine`", "The prior GMM to be adapted (Universal Backgroud Model UBM).")
  .add_parameter("reynolds_adaptation", "bool", "Will use the Reynolds adaptation factor? See Eq (14) from [Reynolds2000]_")
  .add_parameter("relevance_factor", "double", "If set the reynolds_adaptation parameters, will apply the Reynolds Adaptation Factor. See Eq (14) from [Reynolds2000]_")
  .add_parameter("alpha", "double", "Set directly the alpha parameter (Eq (14) from [Reynolds2000]_), ignoring zeroth order statistics as a weighting factor.")
  .add_parameter("other", ":py:class:`bob.learn.misc.MAP_GMMTrainer`", "A MAP_GMMTrainer object to be copied.")
);


static int PyBobLearnMiscMAPGMMTrainer_init_copy(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = MAP_GMMTrainer_doc.kwlist(1);
  PyBobLearnMiscMAPGMMTrainerObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscMAPGMMTrainer_Type, &o)){
    MAP_GMMTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::MAP_GMMTrainer(*o->cxx));
  return 0;
}


static int PyBobLearnMiscMAPGMMTrainer_init_base_trainer(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = MAP_GMMTrainer_doc.kwlist(0);
  
  PyBobLearnMiscGMMBaseTrainerObject* gmm_base_trainer;
  PyBobLearnMiscGMMMachineObject* gmm_machine;
  PyObject* reynolds_adaptation   = 0;
  double alpha = 0.5;
  double relevance_factor = 4.0;


  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|O!dd", kwlist, &PyBobLearnMiscGMMBaseTrainer_Type, &gmm_base_trainer,
                                                                      &PyBobLearnMiscGMMMachine_Type, &gmm_machine,
                                                                      &PyBool_Type, &reynolds_adaptation,
                                                                      &relevance_factor, &alpha )){

    MAP_GMMTrainer_doc.print_usage();
    return -1;
  }
  
  self->cxx.reset(new bob::learn::misc::MAP_GMMTrainer(gmm_base_trainer->cxx, gmm_machine->cxx, f(reynolds_adaptation),relevance_factor, alpha));
  return 0;
}



static int PyBobLearnMiscMAPGMMTrainer_init(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // If the constructor input is GMMBaseTrainer object
  if(PyBobLearnMiscMAPGMMTrainer_Check(args))
    return PyBobLearnMiscMAPGMMTrainer_init_copy(self, args, kwargs);
  else{
    return PyBobLearnMiscMAPGMMTrainer_init_base_trainer(self, args, kwargs);
  }

  BOB_CATCH_MEMBER("cannot create MAP_GMMTrainer", 0)
  return 0;
}


static void PyBobLearnMiscMAPGMMTrainer_delete(PyBobLearnMiscMAPGMMTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnMiscMAPGMMTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscMAPGMMTrainer_Type));
}


static PyObject* PyBobLearnMiscMAPGMMTrainer_RichCompare(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscMAPGMMTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscMAPGMMTrainerObject*>(other);
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


/***** gmm_base_trainer *****/
static auto gmm_base_trainer = bob::extension::VariableDoc(
  "gmm_base_trainer",
  ":py:class:`bob.learn.misc.GMMBaseTrainer`",
  "This class that implements the E-step of the expectation-maximisation algorithm.",
  ""
);
PyObject* PyBobLearnMiscMAPGMMTrainer_getGMMBaseTrainer(PyBobLearnMiscMAPGMMTrainerObject* self, void*){
  BOB_TRY
  
  boost::shared_ptr<bob::learn::misc::GMMBaseTrainer> gmm_base_trainer = self->cxx->getGMMBaseTrainer();

  //Allocating the correspondent python object
  PyBobLearnMiscGMMBaseTrainerObject* retval =
    (PyBobLearnMiscGMMBaseTrainerObject*)PyBobLearnMiscGMMBaseTrainer_Type.tp_alloc(&PyBobLearnMiscGMMBaseTrainer_Type, 0);

  retval->cxx = gmm_base_trainer;

  return Py_BuildValue("O",retval);
  BOB_CATCH_MEMBER("GMMBaseTrainer could not be read", 0)
}
int PyBobLearnMiscMAPGMMTrainer_setGMMBaseTrainer(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBobLearnMiscGMMBaseTrainer_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a :py:class:`bob.learn.misc.GMMBaseTrainer`", Py_TYPE(self)->tp_name, gmm_base_trainer.name());
    return -1;
  }

  PyBobLearnMiscGMMBaseTrainerObject* gmm_base_trainer = 0;
  PyArg_Parse(value, "O!", &PyBobLearnMiscGMMBaseTrainer_Type,&gmm_base_trainer);

  self->cxx->setGMMBaseTrainer(gmm_base_trainer->cxx);

  return 0;
  BOB_CATCH_MEMBER("gmm_base_trainer could not be set", -1)  
}


/***** reynolds_adaptation *****/
static auto reynolds_adaptation = bob::extension::VariableDoc(
  "reynolds_adaptation",
  "bool",
  "Will use the Reynolds adaptation factor? See Eq (14) from [Reynolds2000]_",
  ""
);
PyObject* PyBobLearnMiscMAPGMMTrainer_getReynoldsAdaptation(PyBobLearnMiscMAPGMMTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("O",self->cxx->getReynoldsAdaptation()?Py_True:Py_False);
  BOB_CATCH_MEMBER("reynolds_adaptation could not be read", 0)
}
int PyBobLearnMiscMAPGMMTrainer_setReynoldsAdaptation(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  
  if(!PyBool_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a boolean", Py_TYPE(self)->tp_name, reynolds_adaptation.name());
    return -1;
  }
  
  self->cxx->setReynoldsAdaptation(f(value));
  return 0;
  BOB_CATCH_MEMBER("reynolds_adaptation could not be set", 0)
}


/***** relevance_factor *****/
static auto relevance_factor = bob::extension::VariableDoc(
  "relevance_factor",
  "double",
  "If set the reynolds_adaptation parameters, will apply the Reynolds Adaptation Factor. See Eq (14) from [Reynolds2000]_",
  ""
);
PyObject* PyBobLearnMiscMAPGMMTrainer_getRelevanceFactor(PyBobLearnMiscMAPGMMTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getRelevanceFactor());
  BOB_CATCH_MEMBER("relevance_factor could not be read", 0)
}
int PyBobLearnMiscMAPGMMTrainer_setRelevanceFactor(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  
  if(!PyNumber_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a double", Py_TYPE(self)->tp_name, relevance_factor.name());
    return -1;
  }
  
  self->cxx->setRelevanceFactor(f(value));
  return 0;
  BOB_CATCH_MEMBER("relevance_factor could not be set", 0)
}


/***** alpha *****/
static auto alpha = bob::extension::VariableDoc(
  "alpha",
  "double",
  "Set directly the alpha parameter (Eq (14) from [Reynolds2000]_), ignoring zeroth order statistics as a weighting factor.",
  ""
);
PyObject* PyBobLearnMiscMAPGMMTrainer_getAlpha(PyBobLearnMiscMAPGMMTrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getAlpha());
  BOB_CATCH_MEMBER("alpha could not be read", 0)
}
int PyBobLearnMiscMAPGMMTrainer_setAlpha(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  
  if(!PyNumber_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a double", Py_TYPE(self)->tp_name, alpha.name());
    return -1;
  }
  
  self->cxx->setAlpha(f(value));
  return 0;
  BOB_CATCH_MEMBER("alpha could not be set", 0)
}



static PyGetSetDef PyBobLearnMiscMAPGMMTrainer_getseters[] = { 
  {
    gmm_base_trainer.name(),
    (getter)PyBobLearnMiscMAPGMMTrainer_getGMMBaseTrainer,
    (setter)PyBobLearnMiscMAPGMMTrainer_setGMMBaseTrainer,
    gmm_base_trainer.doc(),
    0
  },
  {
    reynolds_adaptation.name(),
    (getter)PyBobLearnMiscMAPGMMTrainer_getReynoldsAdaptation,
    (setter)PyBobLearnMiscMAPGMMTrainer_setReynoldsAdaptation,
    reynolds_adaptation.doc(),
    0
  },  
  {
    alpha.name(),
    (getter)PyBobLearnMiscMAPGMMTrainer_getAlpha,
    (setter)PyBobLearnMiscMAPGMMTrainer_setAlpha,
    alpha.doc(),
    0
  },
  {
    relevance_factor.name(),
    (getter)PyBobLearnMiscMAPGMMTrainer_getRelevanceFactor,
    (setter)PyBobLearnMiscMAPGMMTrainer_setRelevanceFactor,
    relevance_factor.doc(),
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
.add_prototype("gmm_machine")
.add_parameter("gmm_machine", ":py:class:`bob.learn.misc.GMMMachine`", "GMMMachine Object");
static PyObject* PyBobLearnMiscMAPGMMTrainer_initialize(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnMiscGMMMachineObject* gmm_machine = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMMachine_Type, &gmm_machine)){
    PyErr_Format(PyExc_RuntimeError, "%s.%s. Was not possible to read :py:class:`bob.learn.misc.GMMMachine`", Py_TYPE(self)->tp_name, initialize.name());
    Py_RETURN_NONE;
  }
  self->cxx->initialize(*gmm_machine->cxx);
  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}



/*** mStep ***/
static auto mStep = bob::extension::FunctionDoc(
  "mStep",

   "Performs a maximum a posteriori (MAP) update of the GMM:"  
   "* parameters using the accumulated statistics in :py:class:`bob.learn.misc.GMMBaseTrainer.m_ss` and the" 
   "* parameters of the prior model",
  "",
  true
)
.add_prototype("gmm_machine")
.add_parameter("gmm_machine", ":py:class:`bob.learn.misc.GMMMachine`", "GMMMachine Object");
static PyObject* PyBobLearnMiscMAPGMMTrainer_mStep(PyBobLearnMiscMAPGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = mStep.kwlist(0);

  PyBobLearnMiscGMMMachineObject* gmm_machine;
  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMMachine_Type, &gmm_machine)) Py_RETURN_NONE;

  self->cxx->mStep(*gmm_machine->cxx);

  BOB_CATCH_MEMBER("cannot perform the mStep method", 0)

  Py_RETURN_NONE;
}



static PyMethodDef PyBobLearnMiscMAPGMMTrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnMiscMAPGMMTrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    mStep.name(),
    (PyCFunction)PyBobLearnMiscMAPGMMTrainer_mStep,
    METH_VARARGS|METH_KEYWORDS,
    mStep.doc()
  },
  
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscMAPGMMTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscMAPGMMTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscMAPGMMTrainer_Type.tp_name      = MAP_GMMTrainer_doc.name();
  PyBobLearnMiscMAPGMMTrainer_Type.tp_basicsize = sizeof(PyBobLearnMiscMAPGMMTrainerObject);
  PyBobLearnMiscMAPGMMTrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance
  PyBobLearnMiscMAPGMMTrainer_Type.tp_doc       = MAP_GMMTrainer_doc.doc();

  // set the functions
  PyBobLearnMiscMAPGMMTrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnMiscMAPGMMTrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnMiscMAPGMMTrainer_init);
  PyBobLearnMiscMAPGMMTrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnMiscMAPGMMTrainer_delete);
  PyBobLearnMiscMAPGMMTrainer_Type.tp_richcompare  = reinterpret_cast<richcmpfunc>(PyBobLearnMiscMAPGMMTrainer_RichCompare);
  PyBobLearnMiscMAPGMMTrainer_Type.tp_methods      = PyBobLearnMiscMAPGMMTrainer_methods;
  PyBobLearnMiscMAPGMMTrainer_Type.tp_getset       = PyBobLearnMiscMAPGMMTrainer_getseters;
  //PyBobLearnMiscMAPGMMTrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnMiscMAPGMMTrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscMAPGMMTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscMAPGMMTrainer_Type);
  return PyModule_AddObject(module, "_MAP_GMMTrainer", (PyObject*)&PyBobLearnMiscMAPGMMTrainer_Type) >= 0;
}


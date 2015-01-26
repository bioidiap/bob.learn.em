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
  .add_prototype("gmm_base_trainer","")
  .add_prototype("other","")
  .add_prototype("","")

  .add_parameter("gmm_base_trainer", ":py:class:`bob.learn.misc.GMMBaseTrainer`", "A set GMMBaseTrainer object.")
  .add_parameter("other", ":py:class:`bob.learn.misc.ML_GMMTrainer`", "A ML_GMMTrainer object to be copied.")
);


static int PyBobLearnMiscMLGMMTrainer_init_copy(PyBobLearnMiscMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ML_GMMTrainer_doc.kwlist(1);
  PyBobLearnMiscMLGMMTrainerObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscMLGMMTrainer_Type, &o)){
    ML_GMMTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::ML_GMMTrainer(*o->cxx));
  return 0;
}


static int PyBobLearnMiscMLGMMTrainer_init_base_trainer(PyBobLearnMiscMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ML_GMMTrainer_doc.kwlist(1);
  PyBobLearnMiscGMMBaseTrainerObject* o;
  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMBaseTrainer_Type, &o)){
    ML_GMMTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::ML_GMMTrainer(o->cxx));
  return 0;
}



static int PyBobLearnMiscMLGMMTrainer_init(PyBobLearnMiscMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (nargs==1){ //default initializer ()
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
      return PyBobLearnMiscMLGMMTrainer_init_base_trainer(self, args, kwargs);
    else
      return PyBobLearnMiscMLGMMTrainer_init_copy(self, args, kwargs);
  }
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
    ML_GMMTrainer_doc.print_usage();
    return -1;  
  }

  BOB_CATCH_MEMBER("cannot create GMMBaseTrainer_init_bool", 0)
  return 0;
}


static void PyBobLearnMiscMLGMMTrainer_delete(PyBobLearnMiscMLGMMTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnMiscMLGMMTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscMLGMMTrainer_Type));
}


static PyObject* PyBobLearnMiscMLGMMTrainer_RichCompare(PyBobLearnMiscMLGMMTrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscMLGMMTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscMLGMMTrainerObject*>(other);
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


/***** gmm_base_trainer *****/
static auto gmm_base_trainer = bob::extension::VariableDoc(
  "gmm_base_trainer",
  ":py:class:`bob.learn.misc.GMMBaseTrainer`",
  "This class that implements the E-step of the expectation-maximisation algorithm.",
  ""
);
PyObject* PyBobLearnMiscMLGMMTrainer_getGMMBaseTrainer(PyBobLearnMiscMLGMMTrainerObject* self, void*){
  BOB_TRY
  
  boost::shared_ptr<bob::learn::misc::GMMBaseTrainer> gmm_base_trainer = self->cxx->getGMMBaseTrainer();

  //Allocating the correspondent python object
  PyBobLearnMiscGMMBaseTrainerObject* retval =
    (PyBobLearnMiscGMMBaseTrainerObject*)PyBobLearnMiscGMMBaseTrainer_Type.tp_alloc(&PyBobLearnMiscGMMBaseTrainer_Type, 0);

  retval->cxx = gmm_base_trainer;

  return Py_BuildValue("O",retval);
  BOB_CATCH_MEMBER("GMMBaseTrainer could not be read", 0)
}
int PyBobLearnMiscMLGMMTrainer_setGMMBaseTrainer(PyBobLearnMiscMLGMMTrainerObject* self, PyObject* value, void*){
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


static PyGetSetDef PyBobLearnMiscMLGMMTrainer_getseters[] = { 
  {
    gmm_base_trainer.name(),
    (getter)PyBobLearnMiscMLGMMTrainer_getGMMBaseTrainer,
    (setter)PyBobLearnMiscMLGMMTrainer_setGMMBaseTrainer,
    gmm_base_trainer.doc(),
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
static PyObject* PyBobLearnMiscMLGMMTrainer_initialize(PyBobLearnMiscMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
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
  "Performs a maximum likelihood (ML) update of the GMM parameters "
  "using the accumulated statistics in :py:class:`bob.learn.misc.GMMBaseTrainer.m_ss`",

  "See Section 9.2.2 of Bishop, \"Pattern recognition and machine learning\", 2006",

  true
)
.add_prototype("gmm_machine")
.add_parameter("gmm_machine", ":py:class:`bob.learn.misc.GMMMachine`", "GMMMachine Object");
static PyObject* PyBobLearnMiscMLGMMTrainer_mStep(PyBobLearnMiscMLGMMTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = mStep.kwlist(0);

  PyBobLearnMiscGMMMachineObject* gmm_machine;
  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMMachine_Type, &gmm_machine)) Py_RETURN_NONE;

  self->cxx->mStep(*gmm_machine->cxx);

  BOB_CATCH_MEMBER("cannot perform the mStep method", 0)

  Py_RETURN_NONE;
}



static PyMethodDef PyBobLearnMiscMLGMMTrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnMiscMLGMMTrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    mStep.name(),
    (PyCFunction)PyBobLearnMiscMLGMMTrainer_mStep,
    METH_VARARGS|METH_KEYWORDS,
    mStep.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscMLGMMTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscMLGMMTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscMLGMMTrainer_Type.tp_name      = ML_GMMTrainer_doc.name();
  PyBobLearnMiscMLGMMTrainer_Type.tp_basicsize = sizeof(PyBobLearnMiscMLGMMTrainerObject);
  PyBobLearnMiscMLGMMTrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance
  PyBobLearnMiscMLGMMTrainer_Type.tp_doc       = ML_GMMTrainer_doc.doc();

  // set the functions
  PyBobLearnMiscMLGMMTrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnMiscMLGMMTrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnMiscMLGMMTrainer_init);
  PyBobLearnMiscMLGMMTrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnMiscMLGMMTrainer_delete);
  PyBobLearnMiscMLGMMTrainer_Type.tp_richcompare  = reinterpret_cast<richcmpfunc>(PyBobLearnMiscMLGMMTrainer_RichCompare);
  PyBobLearnMiscMLGMMTrainer_Type.tp_methods      = PyBobLearnMiscMLGMMTrainer_methods;
  PyBobLearnMiscMLGMMTrainer_Type.tp_getset       = PyBobLearnMiscMLGMMTrainer_getseters;
  //PyBobLearnMiscMLGMMTrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnMiscMLGMMTrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscMLGMMTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscMLGMMTrainer_Type);
  return PyModule_AddObject(module, "_ML_GMMTrainer", (PyObject*)&PyBobLearnMiscMLGMMTrainer_Type) >= 0;
}


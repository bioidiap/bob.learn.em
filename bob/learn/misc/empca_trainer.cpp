/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Tue 03 Fev 11:22:00 2015
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto EMPCATrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX "._EMPCATrainer",
  ""

).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a EMPCATrainer",
    "",
    true
  )
  .add_prototype("compute_likelihood","")
  .add_prototype("other","")
  .add_prototype("","")

  .add_parameter("other", ":py:class:`bob.learn.misc.EMPCATrainer`", "A EMPCATrainer object to be copied.")

);


static int PyBobLearnMiscEMPCATrainer_init_copy(PyBobLearnMiscEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = EMPCATrainer_doc.kwlist(1);
  PyBobLearnMiscEMPCATrainerObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscEMPCATrainer_Type, &tt)){
    EMPCATrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::EMPCATrainer(*tt->cxx));
  return 0;
}

static int PyBobLearnMiscEMPCATrainer_init_number(PyBobLearnMiscEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = EMPCATrainer_doc.kwlist(0);
  double convergence_threshold    = 0.0001;
  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &convergence_threshold))
    return -1;

  if(convergence_threshold < 0){
    PyErr_Format(PyExc_TypeError, "convergence_threshold argument must be greater than to zero");
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::EMPCATrainer(convergence_threshold));
  return 0;
}

static int PyBobLearnMiscEMPCATrainer_init(PyBobLearnMiscEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs) {

    case 0:{ //default initializer ()
      self->cxx.reset(new bob::learn::misc::EMPCATrainer());
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

      // If the constructor input is EMPCATrainer object
      if (PyBobLearnMiscEMPCATrainer_Check(arg))
        return PyBobLearnMiscEMPCATrainer_init_copy(self, args, kwargs);
      else if(PyString_Check(arg))
        return PyBobLearnMiscEMPCATrainer_init_number(self, args, kwargs);
    }
    default:{
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0 or 1 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      EMPCATrainer_doc.print_usage();
      return -1;
    }
  }
  BOB_CATCH_MEMBER("cannot create EMPCATrainer", 0)
  return 0;
}


static void PyBobLearnMiscEMPCATrainer_delete(PyBobLearnMiscEMPCATrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnMiscEMPCATrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscEMPCATrainer_Type));
}


static PyObject* PyBobLearnMiscEMPCATrainer_RichCompare(PyBobLearnMiscEMPCATrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscEMPCATrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscEMPCATrainerObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare EMPCATrainer objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/


/***** rng *****/
static auto rng = bob::extension::VariableDoc(
  "rng",
  "str",
  "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.",
  ""
);
PyObject* PyBobLearnMiscEMPCATrainer_getRng(PyBobLearnMiscEMPCATrainerObject* self, void*) {
  BOB_TRY
  //Allocating the correspondent python object
  
  PyBoostMt19937Object* retval =
    (PyBoostMt19937Object*)PyBoostMt19937_Type.tp_alloc(&PyBoostMt19937_Type, 0);

  retval->rng = self->cxx->getRng().get();
  return Py_BuildValue("O", retval);
  BOB_CATCH_MEMBER("Rng method could not be read", 0)
}
int PyBobLearnMiscEMPCATrainer_setRng(PyBobLearnMiscEMPCATrainerObject* self, PyObject* value, void*) {
  BOB_TRY

  if (!PyBoostMt19937_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an PyBoostMt19937_Check", Py_TYPE(self)->tp_name, rng.name());
    return -1;
  }

  PyBoostMt19937Object* boostObject = 0;
  PyBoostMt19937_Converter(value, &boostObject);
  self->cxx->setRng((boost::shared_ptr<boost::mt19937>)boostObject->rng);

  return 0;
  BOB_CATCH_MEMBER("Rng could not be set", 0)
}



static PyGetSetDef PyBobLearnMiscEMPCATrainer_getseters[] = { 
  {
   rng.name(),
   (getter)PyBobLearnMiscEMPCATrainer_getRng,
   (setter)PyBobLearnMiscEMPCATrainer_setRng,
   rng.doc(),
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
  "",
  "",
  true
)
.add_prototype("linear_machine,data")
.add_parameter("linear_machine", ":py:class:`bob.learn.linear.Machine`", "LinearMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnMiscEMPCATrainer_initialize(PyBobLearnMiscEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnLinearMachineObject* linear_machine = 0;
  PyBlitzArrayObject* data                          = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnLinearMachine_Type, &linear_machine,
                                                                 &PyBlitzArray_Converter, &data)) Py_RETURN_NONE;
  auto data_ = make_safe(data);

  self->cxx->initialize(*linear_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));

  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}


/*** eStep ***/
static auto eStep = bob::extension::FunctionDoc(
  "eStep",
  "",
  "",
  true
)
.add_prototype("linear_machine,data")
.add_parameter("linear_machine", ":py:class:`bob.learn.linear.Machine`", "LinearMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnMiscEMPCATrainer_eStep(PyBobLearnMiscEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = eStep.kwlist(0);

  PyBobLearnLinearMachineObject* linear_machine;
  PyBlitzArrayObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnLinearMachine_Type, &linear_machine,
                                                                 &PyBlitzArray_Converter, &data)) Py_RETURN_NONE;
  auto data_ = make_safe(data);

  self->cxx->eStep(*linear_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));


  BOB_CATCH_MEMBER("cannot perform the eStep method", 0)

  Py_RETURN_NONE;
}


/*** mStep ***/
static auto mStep = bob::extension::FunctionDoc(
  "mStep",
  "",
  0,
  true
)
.add_prototype("linear_machine,data")
.add_parameter("linear_machine", ":py:class:`bob.learn.misc.LinearMachine`", "LinearMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnMiscEMPCATrainer_mStep(PyBobLearnMiscEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = mStep.kwlist(0);

  PyBobLearnLinearMachineObject* linear_machine;
  PyBlitzArrayObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnLinearMachine_Type, &linear_machine,
                                                                 &PyBlitzArray_Converter, &data)) Py_RETURN_NONE;
  auto data_ = make_safe(data);

  self->cxx->mStep(*linear_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));


  BOB_CATCH_MEMBER("cannot perform the mStep method", 0)

  Py_RETURN_NONE;
}


/*** computeLikelihood ***/
static auto compute_likelihood = bob::extension::FunctionDoc(
  "compute_likelihood",
  "",
  0,
  true
)
.add_prototype("linear_machine,data")
.add_parameter("linear_machine", ":py:class:`bob.learn.misc.LinearMachine`", "LinearMachine Object");
static PyObject* PyBobLearnMiscEMPCATrainer_compute_likelihood(PyBobLearnMiscEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = compute_likelihood.kwlist(0);

  PyBobLearnLinearMachineObject* linear_machine;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnLinearMachine_Type, &linear_machine)) Py_RETURN_NONE;

  double value = self->cxx->computeLikelihood(*linear_machine->cxx);
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot perform the computeLikelihood method", 0)
}



static PyMethodDef PyBobLearnMiscEMPCATrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnMiscEMPCATrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    eStep.name(),
    (PyCFunction)PyBobLearnMiscEMPCATrainer_eStep,
    METH_VARARGS|METH_KEYWORDS,
    eStep.doc()
  },
  {
    mStep.name(),
    (PyCFunction)PyBobLearnMiscEMPCATrainer_mStep,
    METH_VARARGS|METH_KEYWORDS,
    mStep.doc()
  },
  {
    compute_likelihood.name(),
    (PyCFunction)PyBobLearnMiscEMPCATrainer_compute_likelihood,
    METH_VARARGS|METH_KEYWORDS,
    compute_likelihood.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscEMPCATrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscEMPCATrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscEMPCATrainer_Type.tp_name = EMPCATrainer_doc.name();
  PyBobLearnMiscEMPCATrainer_Type.tp_basicsize = sizeof(PyBobLearnMiscEMPCATrainerObject);
  PyBobLearnMiscEMPCATrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance
  PyBobLearnMiscEMPCATrainer_Type.tp_doc = EMPCATrainer_doc.doc();

  // set the functions
  PyBobLearnMiscEMPCATrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnMiscEMPCATrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnMiscEMPCATrainer_init);
  PyBobLearnMiscEMPCATrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnMiscEMPCATrainer_delete);
  PyBobLearnMiscEMPCATrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscEMPCATrainer_RichCompare);
  PyBobLearnMiscEMPCATrainer_Type.tp_methods = PyBobLearnMiscEMPCATrainer_methods;
  PyBobLearnMiscEMPCATrainer_Type.tp_getset = PyBobLearnMiscEMPCATrainer_getseters;
  PyBobLearnMiscEMPCATrainer_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnMiscEMPCATrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscEMPCATrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscEMPCATrainer_Type);
  return PyModule_AddObject(module, "_EMPCATrainer", (PyObject*)&PyBobLearnMiscEMPCATrainer_Type) >= 0;
}


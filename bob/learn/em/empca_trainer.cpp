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
  BOB_EXT_MODULE_PREFIX ".EMPCATrainer",
   "Trains a :py:class:`bob.learn.linear.Machine` using an Expectation-Maximization algorithm on the given dataset [Bishop1999]_ [Roweis1998]_ \n\n"
    "Notations used are the ones from [Bishop1999]_\n\n"
    "The probabilistic model is given by: :math:`t = Wx + \\mu + \\epsilon`\n\n"
    " - :math:`t` is the observed data (dimension :math:`f`)\n"
    " - :math:`W` is a  projection matrix (dimension :math:`f \\times d`)\n"
    " - :math:`x` is the projected data (dimension :math:`d < f`)\n"
    " - :math:`\\mu` is the mean of the data (dimension :math:`f`)\n"
    " - :math:`\\epsilon` is the noise of the data (dimension :math:`f`) \n"
    " - Gaussian with zero-mean and covariance matrix :math:`\\sigma^2 Id`"

).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",

    "Creates a EMPCATrainer",
    "",
    true
  )
  .add_prototype("other","")
  .add_prototype("","")

  .add_parameter("other", ":py:class:`bob.learn.em.EMPCATrainer`", "A EMPCATrainer object to be copied.")

);


static int PyBobLearnEMEMPCATrainer_init_copy(PyBobLearnEMEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = EMPCATrainer_doc.kwlist(1);
  PyBobLearnEMEMPCATrainerObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMEMPCATrainer_Type, &tt)){
    EMPCATrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::EMPCATrainer(*tt->cxx));
  return 0;
}


static int PyBobLearnEMEMPCATrainer_init(PyBobLearnEMEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs) {

    case 0:{ //default initializer ()
      self->cxx.reset(new bob::learn::em::EMPCATrainer());
      return 0;
    }
    case 1:{
      return PyBobLearnEMEMPCATrainer_init_copy(self, args, kwargs);
    }
    default:{
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0 or 1 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      EMPCATrainer_doc.print_usage();
      return -1;
    }
  }
  BOB_CATCH_MEMBER("cannot create EMPCATrainer", -1)
  return 0;
}


static void PyBobLearnEMEMPCATrainer_delete(PyBobLearnEMEMPCATrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnEMEMPCATrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMEMPCATrainer_Type));
}


static PyObject* PyBobLearnEMEMPCATrainer_RichCompare(PyBobLearnEMEMPCATrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMEMPCATrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMEMPCATrainerObject*>(other);
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


/***** sigma_2 *****/
static auto sigma2 = bob::extension::VariableDoc(
  "sigma2",
  "float",
  "The noise sigma2 of the probabilistic model",
  ""
);
PyObject* PyBobLearnEMEMPCATrainer_getSigma2(PyBobLearnEMEMPCATrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getSigma2());
  BOB_CATCH_MEMBER("sigma2 could not be read", 0)
}
int PyBobLearnEMEMPCATrainer_setSigma2(PyBobLearnEMEMPCATrainerObject* self, PyObject* value, void*){
  BOB_TRY

  if(!PyBob_NumberCheck(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a float", Py_TYPE(self)->tp_name, sigma2.name());
    return -1;
  }

  self->cxx->setSigma2(PyFloat_AS_DOUBLE(value));
  return 0;
  BOB_CATCH_MEMBER("sigma2 could not be set", -1)
}


static PyGetSetDef PyBobLearnEMEMPCATrainer_getseters[] = {
  {
    sigma2.name(),
    (getter)PyBobLearnEMEMPCATrainer_getSigma2,
    (setter)PyBobLearnEMEMPCATrainer_setSigma2,
    sigma2.doc(),
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
.add_prototype("linear_machine, data, [rng]")
.add_parameter("linear_machine", ":py:class:`bob.learn.linear.Machine`", "LinearMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data")
.add_parameter("rng", ":py:class:`bob.core.random.mt19937`", "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.");
static PyObject* PyBobLearnEMEMPCATrainer_initialize(PyBobLearnEMEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnLinearMachineObject* linear_machine = 0;
  PyBlitzArrayObject* data                      = 0;
  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&|O!", kwlist, &PyBobLearnLinearMachine_Type, &linear_machine,
                                                                 &PyBlitzArray_Converter, &data,
                                                                 &PyBoostMt19937_Type, &rng)) return 0;
  auto data_ = make_safe(data);

  if(rng){
    self->cxx->setRng(rng->rng);
  }


  self->cxx->initialize(*linear_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));

  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}


/*** e_step ***/
static auto e_step = bob::extension::FunctionDoc(
  "e_step",
  "",
  "",
  true
)
.add_prototype("linear_machine,data")
.add_parameter("linear_machine", ":py:class:`bob.learn.linear.Machine`", "LinearMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnEMEMPCATrainer_e_step(PyBobLearnEMEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = e_step.kwlist(0);

  PyBobLearnLinearMachineObject* linear_machine;
  PyBlitzArrayObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnLinearMachine_Type, &linear_machine,
                                                                 &PyBlitzArray_Converter, &data)) return 0;
  auto data_ = make_safe(data);

  self->cxx->eStep(*linear_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));


  BOB_CATCH_MEMBER("cannot perform the e_step method", 0)

  Py_RETURN_NONE;
}


/*** m_step ***/
static auto m_step = bob::extension::FunctionDoc(
  "m_step",
  "",
  0,
  true
)
.add_prototype("linear_machine,data")
.add_parameter("linear_machine", ":py:class:`bob.learn.linear.Machine`", "LinearMachine Object")
.add_parameter("data", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnEMEMPCATrainer_m_step(PyBobLearnEMEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = m_step.kwlist(0);

  PyBobLearnLinearMachineObject* linear_machine;
  PyBlitzArrayObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnLinearMachine_Type, &linear_machine,
                                                                 &PyBlitzArray_Converter, &data)) return 0;
  auto data_ = make_safe(data);

  self->cxx->mStep(*linear_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));


  BOB_CATCH_MEMBER("cannot perform the m_step method", 0)

  Py_RETURN_NONE;
}


/*** computeLikelihood ***/
static auto compute_likelihood = bob::extension::FunctionDoc(
  "compute_likelihood",
  "",
  0,
  true
)
.add_prototype("linear_machine")
.add_parameter("linear_machine", ":py:class:`bob.learn.linear.Machine`", "LinearMachine Object");
static PyObject* PyBobLearnEMEMPCATrainer_compute_likelihood(PyBobLearnEMEMPCATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = compute_likelihood.kwlist(0);

  PyBobLearnLinearMachineObject* linear_machine;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnLinearMachine_Type, &linear_machine)) return 0;

  double value = self->cxx->computeLikelihood(*linear_machine->cxx);
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot perform the computeLikelihood method", 0)
}



static PyMethodDef PyBobLearnEMEMPCATrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnEMEMPCATrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    e_step.name(),
    (PyCFunction)PyBobLearnEMEMPCATrainer_e_step,
    METH_VARARGS|METH_KEYWORDS,
    e_step.doc()
  },
  {
    m_step.name(),
    (PyCFunction)PyBobLearnEMEMPCATrainer_m_step,
    METH_VARARGS|METH_KEYWORDS,
    m_step.doc()
  },
  {
    compute_likelihood.name(),
    (PyCFunction)PyBobLearnEMEMPCATrainer_compute_likelihood,
    METH_VARARGS|METH_KEYWORDS,
    compute_likelihood.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMEMPCATrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMEMPCATrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMEMPCATrainer_Type.tp_name = EMPCATrainer_doc.name();
  PyBobLearnEMEMPCATrainer_Type.tp_basicsize = sizeof(PyBobLearnEMEMPCATrainerObject);
  PyBobLearnEMEMPCATrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance
  PyBobLearnEMEMPCATrainer_Type.tp_doc = EMPCATrainer_doc.doc();

  // set the functions
  PyBobLearnEMEMPCATrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnEMEMPCATrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnEMEMPCATrainer_init);
  PyBobLearnEMEMPCATrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnEMEMPCATrainer_delete);
  PyBobLearnEMEMPCATrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMEMPCATrainer_RichCompare);
  PyBobLearnEMEMPCATrainer_Type.tp_methods = PyBobLearnEMEMPCATrainer_methods;
  PyBobLearnEMEMPCATrainer_Type.tp_getset = PyBobLearnEMEMPCATrainer_getseters;
  PyBobLearnEMEMPCATrainer_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMEMPCATrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMEMPCATrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMEMPCATrainer_Type);
  return PyModule_AddObject(module, "EMPCATrainer", (PyObject*)&PyBobLearnEMEMPCATrainer_Type) >= 0;
}

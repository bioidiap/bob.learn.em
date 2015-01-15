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

  .add_parameter("initialization_method", ":py:class:`bob.learn.misc.KMeansTrainer.getInitializationMethod`", "The initialization method of the means")
  .add_parameter("other", ":py:class:`bob.learn.misc.KMeansTrainer`", "A KMeansTrainer object to be copied.")

);


static int PyBobLearnMiscKMeansTrainer_init_copy(PyBobLearnMiscKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = KMeansTrainer_doc.kwlist(1);
  PyBobLearnMiscKMeansTrainerObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscKMeansTrainer_Type, &tt)){
    KMeansTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::KMeansTrainer(*tt->cxx));
  return 0;
}


static int PyBobLearnMiscKMeansTrainer_init(PyBobLearnMiscKMeansTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  //int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);
  
  BOB_CATCH_MEMBER("cannot create KMeansTrainer", 0)
  return 0;
}


static void PyBobLearnMiscKMeansTrainer_delete(PyBobLearnMiscKMeansTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnMiscKMeansTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscKMeansTrainer_Type));
}

/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/


static PyGetSetDef PyBobLearnMiscKMeansTrainer_getseters[] = { 
  {0}  // Sentinel
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/




static PyMethodDef PyBobLearnMiscKMeansTrainer_methods[] = {
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscKMeansTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscKMeansTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscKMeansTrainer_Type.tp_name = KMeansTrainer_doc.name();
  PyBobLearnMiscKMeansTrainer_Type.tp_basicsize = sizeof(PyBobLearnMiscKMeansTrainerObject);
  PyBobLearnMiscKMeansTrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscKMeansTrainer_Type.tp_doc = KMeansTrainer_doc.doc();

  // set the functions
  PyBobLearnMiscKMeansTrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnMiscKMeansTrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnMiscKMeansTrainer_init);
  PyBobLearnMiscKMeansTrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnMiscKMeansTrainer_delete);
  //PyBobLearnMiscKMeansTrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscKMeansTrainer_RichCompare);
  PyBobLearnMiscKMeansTrainer_Type.tp_methods = PyBobLearnMiscKMeansTrainer_methods;
  PyBobLearnMiscKMeansTrainer_Type.tp_getset = PyBobLearnMiscKMeansTrainer_getseters;
  //PyBobLearnMiscGMMMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnMiscGMMMachine_loglikelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscKMeansTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscKMeansTrainer_Type);
  return PyModule_AddObject(module, "KMeansTrainer", (PyObject*)&PyBobLearnMiscKMeansTrainer_Type) >= 0;
}


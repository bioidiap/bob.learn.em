/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Wed 04 Feb 14:15:00 2015
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

template <int N>
int list_as_vector(PyObject* list, std::vector<blitz::Array<double,N> >& vec)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++)
  {
    PyBlitzArrayObject* blitz_object; 
    if (!PyArg_Parse(PyList_GetItem(list, i), "O&", &PyBlitzArray_Converter, &blitz_object)){
      PyErr_Format(PyExc_RuntimeError, "Expected numpy array object");
      return -1;
    }
    auto blitz_object_ = make_safe(blitz_object);
    vec.push_back(*PyBlitzArrayCxx_AsBlitz<double,N>(blitz_object));
  }
  return 0;
}


static auto PLDATrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".PLDATrainer",
  "This class can be used to train the :math:`$F$`, :math:`$G$ and "
  " :math:`$\\Sigma$` matrices and the mean vector :math:`$\\mu$` of a PLDA model.",
  "References: [ElShafey2014,PrinceElder2007,LiFu2012]",
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Default constructor.\n Initializes a new PLDA trainer. The "
    "training stage will place the resulting components in the "
    "PLDABase.",
    "",
    true
  )
  .add_prototype("use_sum_second_order","")
  .add_prototype("other","")
  .add_prototype("","")

  .add_parameter("other", ":py:class:`bob.learn.misc.PLDATrainer`", "A PLDATrainer object to be copied.")
  .add_parameter("use_sum_second_order", "bool", "")
);

static int PyBobLearnMiscPLDATrainer_init_copy(PyBobLearnMiscPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDATrainer_doc.kwlist(1);
  PyBobLearnMiscPLDATrainerObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscPLDATrainer_Type, &o)){
    PLDATrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::PLDATrainer(*o->cxx));
  return 0;
}


static int PyBobLearnMiscPLDATrainer_init_bool(PyBobLearnMiscPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDATrainer_doc.kwlist(0);
  PyObject* use_sum_second_order;

  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBool_Type, &use_sum_second_order))
    return -1;

  self->cxx.reset(new bob::learn::misc::PLDATrainer(f(use_sum_second_order)));
  return 0;
}


static int PyBobLearnMiscPLDATrainer_init(PyBobLearnMiscPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch(nargs){
    case 0:{
      self->cxx.reset(new bob::learn::misc::PLDATrainer());
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
      
      if(PyBobLearnMiscPLDATrainer_Check(arg))
        // If the constructor input is PLDATrainer object
        return PyBobLearnMiscPLDATrainer_init_copy(self, args, kwargs);
      else
        return PyBobLearnMiscPLDATrainer_init_bool(self, args, kwargs);

    }
    default:{
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires only 0 or 1 argument, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      PLDATrainer_doc.print_usage();
      return -1;
    }
  }
  BOB_CATCH_MEMBER("cannot create PLDATrainer", 0)
  return 0;
}


static void PyBobLearnMiscPLDATrainer_delete(PyBobLearnMiscPLDATrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnMiscPLDATrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscPLDATrainer_Type));
}


static PyObject* PyBobLearnMiscPLDATrainer_RichCompare(PyBobLearnMiscPLDATrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscPLDATrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscPLDATrainerObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare PLDATrainer objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto z_second_order = bob::extension::VariableDoc(
  "z_second_order",
  "array_like <float, 3D>",
  "",
  ""
);
PyObject* PyBobLearnMiscPLDATrainer_get_z_second_order(PyBobLearnMiscPLDATrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZSecondOrder());
  BOB_CATCH_MEMBER("z_second_order could not be read", 0)
}


static auto z_second_order_sum = bob::extension::VariableDoc(
  "z_second_order_sum",
  "array_like <float, 2D>",
  "",
  ""
);
PyObject* PyBobLearnMiscPLDATrainer_get_z_second_order_sum(PyBobLearnMiscPLDATrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZSecondOrderSum());
  BOB_CATCH_MEMBER("z_second_order_sum could not be read", 0)
}


static auto z_first_order = bob::extension::VariableDoc(
  "z_first_order",
  "array_like <float, 2D>",
  "",
  ""
);
PyObject* PyBobLearnMiscPLDATrainer_get_z_first_order(PyBobLearnMiscPLDATrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZFirstOrder());
  BOB_CATCH_MEMBER("z_first_order could not be read", 0)
}




static PyGetSetDef PyBobLearnMiscPLDATrainer_getseters[] = { 
  {
   z_first_order.name(),
   (getter)PyBobLearnMiscPLDATrainer_get_z_first_order,
   0,
   z_first_order.doc(),
   0
  },
  {
   z_second_order_sum.name(),
   (getter)PyBobLearnMiscPLDATrainer_get_z_second_order_sum,
   0,
   z_second_order_sum.doc(),
   0
  },
  {
   z_second_order.name(),
   (getter)PyBobLearnMiscPLDATrainer_get_z_second_order,
   0,
   z_second_order.doc(),
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
.add_prototype("plda_base,data")
.add_parameter("plda_base", ":py:class:`bob.learn.misc.PLDABase`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnMiscPLDATrainer_initialize(PyBobLearnMiscPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnMiscPLDABaseObject* plda_base = 0;
  PyObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnMiscPLDABase_Type, &plda_base,
                                                                 &PyList_Type, &data)) Py_RETURN_NONE;

  std::vector<blitz::Array<double,2> > data_vector;
  if(list_as_vector(data ,data_vector)==0)
    self->cxx->initialize(*plda_machine->cxx, data_vector);

  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}


/*** e_step ***/
static auto e_step = bob::extension::FunctionDoc(
  "e_step",
  "e_step before the EM steps",
  "",
  true
)
.add_prototype("plda_base,data")
.add_parameter("plda_base", ":py:class:`bob.learn.misc.PLDABase`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnMiscPLDATrainer_e_step(PyBobLearnMiscPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = e_step.kwlist(0);

  PyBobLearnMiscPLDABaseObject* plda_base = 0;
  PyObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnMiscPLDABase_Type, &plda_base,
                                                                 &PyList_Type, &data)) Py_RETURN_NONE;

  std::vector<blitz::Array<double,2> > data_vector;
  if(list_as_vector(data ,data_vector)==0)
    self->cxx->e_step(*plda_machine->cxx, data_vector);

  BOB_CATCH_MEMBER("cannot perform the e_step method", 0)

  Py_RETURN_NONE;
}


/*** m_step ***/
static auto m_step = bob::extension::FunctionDoc(
  "m_step",
  "m_step before the EM steps",
  "",
  true
)
.add_prototype("plda_base,data")
.add_parameter("plda_base", ":py:class:`bob.learn.misc.PLDABase`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnMiscPLDATrainer_m_step(PyBobLearnMiscPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = m_step.kwlist(0);

  PyBobLearnMiscPLDABaseObject* plda_base = 0;
  PyObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnMiscPLDABase_Type, &plda_base,
                                                                 &PyList_Type, &data)) Py_RETURN_NONE;

  std::vector<blitz::Array<double,2> > data_vector;
  if(list_as_vector(data ,data_vector)==0)
    self->cxx->m_step(*plda_machine->cxx, data_vector);

  BOB_CATCH_MEMBER("cannot perform the m_step method", 0)

  Py_RETURN_NONE;
}


/*** finalize ***/
static auto finalize = bob::extension::FunctionDoc(
  "finalize",
  "finalize before the EM steps",
  "",
  true
)
.add_prototype("plda_base,data")
.add_parameter("plda_base", ":py:class:`bob.learn.misc.PLDABase`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnMiscPLDATrainer_finalize(PyBobLearnMiscPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = finalize.kwlist(0);

  PyBobLearnMiscPLDABaseObject* plda_base = 0;
  PyObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnMiscPLDABase_Type, &plda_base,
                                                                 &PyList_Type, &data)) Py_RETURN_NONE;

  std::vector<blitz::Array<double,2> > data_vector;
  if(list_as_vector(data ,data_vector)==0)
    self->cxx->finalize(*plda_machine->cxx, data_vector);

  BOB_CATCH_MEMBER("cannot perform the finalize method", 0)

  Py_RETURN_NONE;
}



/*** enrol ***/
static auto enrol = bob::extension::FunctionDoc(
  "enrol",
  "Main procedure for enrolling a PLDAMachine",
  "",
  true
)
.add_prototype("plda_machine,data")
.add_parameter("plda_machine", ":py:class:`bob.learn.misc.PLDAMachine`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnMiscPLDATrainer_finalize(PyBobLearnMiscPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = finalize.kwlist(0);

  PyBobLearnMiscPLDAMachineObject* plda_machine = 0;
  PyBlitzArrayObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnMiscPLDAMachine_Type, &plda_machine,
                                                                 &PyBlitzArray_Converter, &data)) Py_RETURN_NONE;

  auto data_ = make_safe(data);
  self->cxx->enrol(*plda_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));

  BOB_CATCH_MEMBER("cannot perform the enrol method", 0)

  Py_RETURN_NONE;
}


static PyMethodDef PyBobLearnMiscPLDATrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnMiscPLDATrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    e_step.name(),
    (PyCFunction)PyBobLearnMiscPLDATrainer_e_step,
    METH_VARARGS|METH_KEYWORDS,
    e_step.doc()
  },
  {
    m_step.name(),
    (PyCFunction)PyBobLearnMiscPLDATrainer_m_step,
    METH_VARARGS|METH_KEYWORDS,
    m_step.doc()
  },
  {
    enrol.name(),
    (PyCFunction)PyBobLearnMiscPLDATrainer_enrol,
    METH_VARARGS|METH_KEYWORDS,
    enrol.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscPLDATrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscPLDATrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscPLDATrainer_Type.tp_name      = PLDATrainer_doc.name();
  PyBobLearnMiscPLDATrainer_Type.tp_basicsize = sizeof(PyBobLearnMiscPLDATrainerObject);
  PyBobLearnMiscPLDATrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance;
  PyBobLearnMiscPLDATrainer_Type.tp_doc       = PLDATrainer_doc.doc();

  // set the functions
  PyBobLearnMiscPLDATrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnMiscPLDATrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnMiscPLDATrainer_init);
  PyBobLearnMiscPLDATrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnMiscPLDATrainer_delete);
  PyBobLearnMiscPLDATrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscPLDATrainer_RichCompare);
  PyBobLearnMiscPLDATrainer_Type.tp_methods      = PyBobLearnMiscPLDATrainer_methods;
  PyBobLearnMiscPLDATrainer_Type.tp_getset       = PyBobLearnMiscPLDATrainer_getseters;
  //PyBobLearnMiscPLDATrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnMiscPLDATrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscPLDATrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscPLDATrainer_Type);
  return PyModule_AddObject(module, "_PLDATrainer", (PyObject*)&PyBobLearnMiscPLDATrainer_Type) >= 0;
}


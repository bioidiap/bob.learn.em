/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Mon 02 Fev 20:20:00 2015
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

static int extract_GMMStats_1d(PyObject *list,
                             std::vector<boost::shared_ptr<bob::learn::em::GMMStats> >& training_data)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++){

    PyBobLearnEMGMMStatsObject* stats;
    if (!PyArg_Parse(PyList_GetItem(list, i), "O!", &PyBobLearnEMGMMStats_Type, &stats)){
      PyErr_Format(PyExc_RuntimeError, "Expected GMMStats objects");
      return -1;
    }
    training_data.push_back(stats->cxx);
  }
  return 0;
}

static int extract_GMMStats_2d(PyObject *list,
                             std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > >& training_data)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++)
  {
    PyObject* another_list;
    if(!PyArg_Parse(PyList_GetItem(list, i), "O!", &PyList_Type, &another_list)){
      PyErr_Format(PyExc_RuntimeError, "Expected a list of lists of GMMStats objects [[GMMStats,GMMStats],[GMMStats,GMMStats].....[GMMStats,GMMStats]]");
      return -1;
    }

    std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > another_training_data;
    for (int j=0; j<PyList_GET_SIZE(another_list); j++){

      PyBobLearnEMGMMStatsObject* stats;
      if (!PyArg_Parse(PyList_GetItem(another_list, j), "O!", &PyBobLearnEMGMMStats_Type, &stats)){
        PyErr_Format(PyExc_RuntimeError, "Expected GMMStats objects");
        return -1;
      }
      another_training_data.push_back(stats->cxx);
    }
    training_data.push_back(another_training_data);
  }
  return 0;
}

template <int N>
static PyObject* vector_as_list(const std::vector<blitz::Array<double,N> >& vec)
{
  PyObject* list = PyList_New(vec.size());
  for(size_t i=0; i<vec.size(); i++){
    blitz::Array<double,N> numpy_array = vec[i];
    PyObject* numpy_py_object = PyBlitzArrayCxx_AsNumpy(numpy_array);
    PyList_SET_ITEM(list, i, numpy_py_object);
  }
  return list;
}

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



static auto ISVTrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".ISVTrainer",
  "ISVTrainer"
  "References: [Vogt2008,McCool2013]",
  ""
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructor. Builds a new ISVTrainer",
    "",
    true
  )
  .add_prototype("relevance_factor","")
  .add_prototype("other","")
  .add_prototype("","")
  .add_parameter("other", ":py:class:`bob.learn.em.ISVTrainer`", "A ISVTrainer object to be copied.")
  .add_parameter("relevance_factor", "float", "")
);


static int PyBobLearnEMISVTrainer_init_copy(PyBobLearnEMISVTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVTrainer_doc.kwlist(1);
  PyBobLearnEMISVTrainerObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMISVTrainer_Type, &o)){
    ISVTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::ISVTrainer(*o->cxx));
  return 0;
}


static int PyBobLearnEMISVTrainer_init_number(PyBobLearnEMISVTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVTrainer_doc.kwlist(0);
  double relevance_factor      = 4.;

  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &relevance_factor))
    return -1;

  if(relevance_factor < 0){
    PyErr_Format(PyExc_TypeError, "gaussians argument must be greater than zero");
    return -1;
  }

  self->cxx.reset(new bob::learn::em::ISVTrainer(relevance_factor));
  return 0;
}


static int PyBobLearnEMISVTrainer_init(PyBobLearnEMISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch(nargs){
    case 0:{
      self->cxx.reset(new bob::learn::em::ISVTrainer());
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

      if(PyBobLearnEMISVTrainer_Check(arg))
        // If the constructor input is ISVTrainer object
        return PyBobLearnEMISVTrainer_init_copy(self, args, kwargs);
      else
        return PyBobLearnEMISVTrainer_init_number(self, args, kwargs);

    }
    default:{
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires only 0 or 1 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      ISVTrainer_doc.print_usage();
      return -1;
    }
  }
  BOB_CATCH_MEMBER("cannot create ISVTrainer", -1)
  return 0;
}


static void PyBobLearnEMISVTrainer_delete(PyBobLearnEMISVTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnEMISVTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMISVTrainer_Type));
}


static PyObject* PyBobLearnEMISVTrainer_RichCompare(PyBobLearnEMISVTrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMISVTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMISVTrainerObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare ISVTrainer objects", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto acc_u_a1 = bob::extension::VariableDoc(
  "acc_u_a1",
  "array_like <float, 3D>",
  "Accumulator updated during the E-step",
  ""
);
PyObject* PyBobLearnEMISVTrainer_get_acc_u_a1(PyBobLearnEMISVTrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getAccUA1());
  BOB_CATCH_MEMBER("acc_u_a1 could not be read", 0)
}
int PyBobLearnEMISVTrainer_set_acc_u_a1(PyBobLearnEMISVTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 3D array of floats", Py_TYPE(self)->tp_name, acc_u_a1.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, acc_u_a1.name());
    return -1;
  }

  if (input->ndim != 3){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 3D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, acc_u_a1.name());
    return -1;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getAccUA1().extent(0) && input->shape[1] != (Py_ssize_t)self->cxx->getAccUA1().extent(1) && input->shape[2] != (Py_ssize_t)self->cxx->getAccUA1().extent(2)) {
    PyErr_Format(PyExc_TypeError, "`%s' 3D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getAccUA1().extent(0), (Py_ssize_t)self->cxx->getAccUA1().extent(1), (Py_ssize_t)self->cxx->getAccUA1().extent(2), (Py_ssize_t)input->shape[0], (Py_ssize_t)input->shape[1], (Py_ssize_t)input->shape[2], acc_u_a1.name());
    return -1;
  }


  auto b = PyBlitzArrayCxx_AsBlitz<double,3>(input, "acc_u_a1");
  if (!b) return -1;
  self->cxx->setAccUA1(*b);
  return 0;
  BOB_CATCH_MEMBER("acc_u_a1 could not be set", -1)
}


static auto acc_u_a2 = bob::extension::VariableDoc(
  "acc_u_a2",
  "array_like <float, 2D>",
  "Accumulator updated during the E-step",
  ""
);
PyObject* PyBobLearnEMISVTrainer_get_acc_u_a2(PyBobLearnEMISVTrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getAccUA2());
  BOB_CATCH_MEMBER("acc_u_a2 could not be read", 0)
}
int PyBobLearnEMISVTrainer_set_acc_u_a2(PyBobLearnEMISVTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, acc_u_a2.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, acc_u_a2.name());
    return -1;
  }

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, acc_u_a2.name());
    return -1;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getAccUA2().extent(0) && input->shape[1] != (Py_ssize_t)self->cxx->getAccUA2().extent(1)) {
    PyErr_Format(PyExc_TypeError, "`%s' 3D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getAccUA2().extent(0), (Py_ssize_t)self->cxx->getAccUA2().extent(1), input->shape[0], input->shape[1], acc_u_a2.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "acc_u_a2");
  if (!b) return -1;
  self->cxx->setAccUA2(*b);
  return 0;
  BOB_CATCH_MEMBER("acc_u_a2 could not be set", -1)
}


static auto __X__ = bob::extension::VariableDoc(
  "__X__",
  "list",
  "",
  ""
);
PyObject* PyBobLearnEMISVTrainer_get_X(PyBobLearnEMISVTrainerObject* self, void*){
  BOB_TRY
  return vector_as_list(self->cxx->getX());
  BOB_CATCH_MEMBER("__X__ could not be read", 0)
}
int PyBobLearnEMISVTrainer_set_X(PyBobLearnEMISVTrainerObject* self, PyObject* value, void*){
  BOB_TRY

  // Parses input arguments in a single shot
  if (!PyList_Check(value)){
    PyErr_Format(PyExc_TypeError, "Expected a list in `%s'", __X__.name());
    return -1;
  }

  std::vector<blitz::Array<double,2> > data;
  if(list_as_vector(value ,data)==0){
    self->cxx->setX(data);
  }

  return 0;
  BOB_CATCH_MEMBER("__X__ could not be written", -1)
}


static auto __Z__ = bob::extension::VariableDoc(
  "__Z__",
  "list",
  "",
  ""
);
PyObject* PyBobLearnEMISVTrainer_get_Z(PyBobLearnEMISVTrainerObject* self, void*){
  BOB_TRY
  return vector_as_list(self->cxx->getZ());
  BOB_CATCH_MEMBER("__Z__ could not be read", 0)
}
int PyBobLearnEMISVTrainer_set_Z(PyBobLearnEMISVTrainerObject* self, PyObject* value, void*){
  BOB_TRY

  // Parses input arguments in a single shot
  if (!PyList_Check(value)){
    PyErr_Format(PyExc_TypeError, "Expected a list in `%s'", __Z__.name());
    return -1;
  }

  std::vector<blitz::Array<double,1> > data;
  if(list_as_vector(value ,data)==0){
    self->cxx->setZ(data);
  }

  return 0;
  BOB_CATCH_MEMBER("__Z__ could not be written", -1)
}


static PyGetSetDef PyBobLearnEMISVTrainer_getseters[] = {
  {
   acc_u_a1.name(),
   (getter)PyBobLearnEMISVTrainer_get_acc_u_a1,
   (setter)PyBobLearnEMISVTrainer_set_acc_u_a1,
   acc_u_a1.doc(),
   0
  },
  {
   acc_u_a2.name(),
   (getter)PyBobLearnEMISVTrainer_get_acc_u_a2,
   (setter)PyBobLearnEMISVTrainer_set_acc_u_a2,
   acc_u_a2.doc(),
   0
  },
  {
   __X__.name(),
   (getter)PyBobLearnEMISVTrainer_get_X,
   (setter)PyBobLearnEMISVTrainer_set_X,
   __X__.doc(),
   0
  },
  {
   __Z__.name(),
   (getter)PyBobLearnEMISVTrainer_get_Z,
   (setter)PyBobLearnEMISVTrainer_set_Z,
   __Z__.doc(),
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
.add_prototype("isv_base, stats, [rng]")
.add_parameter("isv_base", ":py:class:`bob.learn.em.ISVBase`", "ISVBase Object")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "GMMStats Object")
.add_parameter("rng", ":py:class:`bob.core.random.mt19937`", "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.");
static PyObject* PyBobLearnEMISVTrainer_initialize(PyBobLearnEMISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnEMISVBaseObject* isv_base = 0;
  PyObject* stats = 0;
  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|O!", kwlist, &PyBobLearnEMISVBase_Type, &isv_base,
                                                                 &PyList_Type, &stats,
                                                                 &PyBoostMt19937_Type, &rng)) return 0;

  if(rng){
    self->cxx->setRng(rng->rng);
  }

  std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > > training_data;
  if(extract_GMMStats_2d(stats ,training_data)==0)
    self->cxx->initialize(*isv_base->cxx, training_data);
  else
    return 0;

  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}


/*** e_step ***/
static auto e_step = bob::extension::FunctionDoc(
  "e_step",
  "Call the e-step procedure (for the U subspace).",
  "",
  true
)
.add_prototype("isv_base, stats")
.add_parameter("isv_base", ":py:class:`bob.learn.em.ISVBase`", "ISVBase Object")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "GMMStats Object");
static PyObject* PyBobLearnEMISVTrainer_e_step(PyBobLearnEMISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // Parses input arguments in a single shot
  char** kwlist = e_step.kwlist(0);

  PyBobLearnEMISVBaseObject* isv_base = 0;
  PyObject* stats = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnEMISVBase_Type, &isv_base,
                                                                 &PyList_Type, &stats)) return 0;

  std::vector<std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > > training_data;
  if(extract_GMMStats_2d(stats ,training_data)==0)
    self->cxx->eStep(*isv_base->cxx, training_data);
  else
    return 0;

  BOB_CATCH_MEMBER("cannot perform the e_step method", 0)

  Py_RETURN_NONE;
}


/*** m_step ***/
static auto m_step = bob::extension::FunctionDoc(
  "m_step",
  "Call the m-step procedure (for the U subspace).",
  "",
  true
)
.add_prototype("isv_base, [stats]")
.add_parameter("isv_base", ":py:class:`bob.learn.em.ISVBase`", "ISVBase Object")
.add_parameter("stats", "object", "Ignored.");
static PyObject* PyBobLearnEMISVTrainer_m_step(PyBobLearnEMISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // Parses input arguments in a single shot
  char** kwlist = m_step.kwlist(0);

  PyBobLearnEMISVBaseObject* isv_base = 0;
  PyObject* stats;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", kwlist, &PyBobLearnEMISVBase_Type, &isv_base,
                                                                 &stats)) return 0;

  self->cxx->mStep(*isv_base->cxx);

  BOB_CATCH_MEMBER("cannot perform the m_step method", 0)

  Py_RETURN_NONE;
}



/*** enroll ***/
static auto enroll = bob::extension::FunctionDoc(
  "enroll",
  "",
  "",
  true
)
.add_prototype("isv_machine, features, n_iter")
.add_parameter("isv_machine", ":py:class:`bob.learn.em.ISVMachine`", "ISVMachine Object")
.add_parameter("features", "list(:py:class:`bob.learn.em.GMMStats`)`", "")
.add_parameter("n_iter", "int", "Number of iterations");
static PyObject* PyBobLearnEMISVTrainer_enroll(PyBobLearnEMISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // Parses input arguments in a single shot
  char** kwlist = enroll.kwlist(0);

  PyBobLearnEMISVMachineObject* isv_machine = 0;
  PyObject* stats = 0;
  int n_iter = 1;


  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!i", kwlist, &PyBobLearnEMISVMachine_Type, &isv_machine,
                                                                  &PyList_Type, &stats, &n_iter)) return 0;

  std::vector<boost::shared_ptr<bob::learn::em::GMMStats> > training_data;
  if(extract_GMMStats_1d(stats ,training_data)==0)
    self->cxx->enroll(*isv_machine->cxx, training_data, n_iter);
  else
    return 0;

  BOB_CATCH_MEMBER("cannot perform the enroll method", 0)

  Py_RETURN_NONE;
}



static PyMethodDef PyBobLearnEMISVTrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnEMISVTrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    e_step.name(),
    (PyCFunction)PyBobLearnEMISVTrainer_e_step,
    METH_VARARGS|METH_KEYWORDS,
    e_step.doc()
  },
  {
    m_step.name(),
    (PyCFunction)PyBobLearnEMISVTrainer_m_step,
    METH_VARARGS|METH_KEYWORDS,
    m_step.doc()
  },
  {
    enroll.name(),
    (PyCFunction)PyBobLearnEMISVTrainer_enroll,
    METH_VARARGS|METH_KEYWORDS,
    enroll.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMISVTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMISVTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMISVTrainer_Type.tp_name      = ISVTrainer_doc.name();
  PyBobLearnEMISVTrainer_Type.tp_basicsize = sizeof(PyBobLearnEMISVTrainerObject);
  PyBobLearnEMISVTrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance;
  PyBobLearnEMISVTrainer_Type.tp_doc       = ISVTrainer_doc.doc();

  // set the functions
  PyBobLearnEMISVTrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnEMISVTrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnEMISVTrainer_init);
  PyBobLearnEMISVTrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnEMISVTrainer_delete);
  PyBobLearnEMISVTrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMISVTrainer_RichCompare);
  PyBobLearnEMISVTrainer_Type.tp_methods      = PyBobLearnEMISVTrainer_methods;
  PyBobLearnEMISVTrainer_Type.tp_getset       = PyBobLearnEMISVTrainer_getseters;
  //PyBobLearnEMISVTrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnEMISVTrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMISVTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMISVTrainer_Type);
  return PyModule_AddObject(module, "ISVTrainer", (PyObject*)&PyBobLearnEMISVTrainer_Type) >= 0;
}

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
                             std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> >& training_data)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++){
  
    PyBobLearnMiscGMMStatsObject* stats;
    if (!PyArg_Parse(PyList_GetItem(list, i), "O!", &PyBobLearnMiscGMMStats_Type, &stats)){
      PyErr_Format(PyExc_RuntimeError, "Expected GMMStats objects");
      return -1;
    }
    training_data.push_back(stats->cxx);
  }
  return 0;
}

static int extract_GMMStats_2d(PyObject *list,
                             std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > >& training_data)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++)
  {
    PyObject* another_list;
    PyArg_Parse(PyList_GetItem(list, i), "O!", &PyList_Type, &another_list);

    std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > another_training_data;
    for (int j=0; j<PyList_GET_SIZE(another_list); j++){

      PyBobLearnMiscGMMStatsObject* stats;
      if (!PyArg_Parse(PyList_GetItem(another_list, j), "O!", &PyBobLearnMiscGMMStats_Type, &stats)){
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
  .add_prototype("relevance_factor,convergence_threshold","")
  .add_prototype("other","")
  .add_prototype("","")
  .add_parameter("other", ":py:class:`bob.learn.misc.ISVTrainer`", "A ISVTrainer object to be copied.")
  .add_parameter("relevance_factor", "double", "")
  .add_parameter("convergence_threshold", "double", "")
);


static int PyBobLearnMiscISVTrainer_init_copy(PyBobLearnMiscISVTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVTrainer_doc.kwlist(1);
  PyBobLearnMiscISVTrainerObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscISVTrainer_Type, &o)){
    ISVTrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::ISVTrainer(*o->cxx));
  return 0;
}


static int PyBobLearnMiscISVTrainer_init_number(PyBobLearnMiscISVTrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVTrainer_doc.kwlist(0);
  double relevance_factor      = 4.;
  double convergence_threshold = 0.001;
  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dd", kwlist, &relevance_factor, &convergence_threshold))
    return -1;

  if(relevance_factor < 0){
    PyErr_Format(PyExc_TypeError, "gaussians argument must be greater than zero");
    return -1;
  }

  if(convergence_threshold < 0){
    PyErr_Format(PyExc_TypeError, "convergence_threshold argument must be greater than zero");
    return -1;
   }

  self->cxx.reset(new bob::learn::misc::ISVTrainer(relevance_factor, convergence_threshold));
  return 0;
}


static int PyBobLearnMiscISVTrainer_init(PyBobLearnMiscISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch(nargs){
    case 0:{
      self->cxx.reset(new bob::learn::misc::ISVTrainer());
      return 0;
    }
    case 1:{
      // If the constructor input is ISVTrainer object
      return PyBobLearnMiscISVTrainer_init_copy(self, args, kwargs);
    }
    case 2:{
      // If the constructor input is ISVTrainer object
      return PyBobLearnMiscISVTrainer_init_number(self, args, kwargs);
    }
    default:{
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires only 0, 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      ISVTrainer_doc.print_usage();
      return -1;
    }
  }
  BOB_CATCH_MEMBER("cannot create ISVTrainer", 0)
  return 0;
}


static void PyBobLearnMiscISVTrainer_delete(PyBobLearnMiscISVTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnMiscISVTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscISVTrainer_Type));
}


static PyObject* PyBobLearnMiscISVTrainer_RichCompare(PyBobLearnMiscISVTrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscISVTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscISVTrainerObject*>(other);
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
PyObject* PyBobLearnMiscISVTrainer_get_acc_u_a1(PyBobLearnMiscISVTrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getAccUA1());
  BOB_CATCH_MEMBER("acc_u_a1 could not be read", 0)
}
int PyBobLearnMiscISVTrainer_set_acc_u_a1(PyBobLearnMiscISVTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 3D array of floats", Py_TYPE(self)->tp_name, acc_u_a1.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,3>(o, "acc_u_a1");
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
PyObject* PyBobLearnMiscISVTrainer_get_acc_u_a2(PyBobLearnMiscISVTrainerObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getAccUA2());
  BOB_CATCH_MEMBER("acc_u_a2 could not be read", 0)
}
int PyBobLearnMiscISVTrainer_set_acc_u_a2(PyBobLearnMiscISVTrainerObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, acc_u_a2.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(o, "acc_u_a2");
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
PyObject* PyBobLearnMiscISVTrainer_get_X(PyBobLearnMiscISVTrainerObject* self, void*){
  BOB_TRY
  return vector_as_list(self->cxx->getX());
  BOB_CATCH_MEMBER("__X__ could not be read", 0)
}
int PyBobLearnMiscISVTrainer_set_X(PyBobLearnMiscISVTrainerObject* self, PyObject* value, void*){
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
  BOB_CATCH_MEMBER("__X__ could not be written", 0)
}


static auto __Z__ = bob::extension::VariableDoc(
  "__Z__",
  "list",
  "",
  ""
);
PyObject* PyBobLearnMiscISVTrainer_get_Z(PyBobLearnMiscISVTrainerObject* self, void*){
  BOB_TRY
  return vector_as_list(self->cxx->getZ());
  BOB_CATCH_MEMBER("__Z__ could not be read", 0)
}
int PyBobLearnMiscISVTrainer_set_Z(PyBobLearnMiscISVTrainerObject* self, PyObject* value, void*){
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
  BOB_CATCH_MEMBER("__Z__ could not be written", 0)
}




static PyGetSetDef PyBobLearnMiscISVTrainer_getseters[] = { 
  {
   acc_u_a1.name(),
   (getter)PyBobLearnMiscISVTrainer_get_acc_u_a1,
   (setter)PyBobLearnMiscISVTrainer_get_acc_u_a1,
   acc_u_a1.doc(),
   0
  },
  {
   acc_u_a2.name(),
   (getter)PyBobLearnMiscISVTrainer_get_acc_u_a2,
   (setter)PyBobLearnMiscISVTrainer_get_acc_u_a2,
   acc_u_a2.doc(),
   0
  },
  {
   __X__.name(),
   (getter)PyBobLearnMiscISVTrainer_get_X,
   (setter)PyBobLearnMiscISVTrainer_set_X,
   __X__.doc(),
   0
  },
  {
   __Z__.name(),
   (getter)PyBobLearnMiscISVTrainer_get_Z,
   (setter)PyBobLearnMiscISVTrainer_set_Z,
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
.add_prototype("isv_base,stats")
.add_parameter("isv_base", ":py:class:`bob.learn.misc.ISVBase`", "ISVBase Object")
.add_parameter("stats", ":py:class:`bob.learn.misc.GMMStats`", "GMMStats Object");
static PyObject* PyBobLearnMiscISVTrainer_initialize(PyBobLearnMiscISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnMiscISVBaseObject* isv_base = 0;
  PyObject* stats = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnMiscISVBase_Type, &isv_base,
                                                                 &PyList_Type, &stats)) Py_RETURN_NONE;

  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  if(extract_GMMStats_2d(stats ,training_data)==0)
    self->cxx->initialize(*isv_base->cxx, training_data);

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
.add_prototype("isv_base,stats")
.add_parameter("isv_base", ":py:class:`bob.learn.misc.ISVBase`", "ISVBase Object")
.add_parameter("stats", ":py:class:`bob.learn.misc.GMMStats`", "GMMStats Object");
static PyObject* PyBobLearnMiscISVTrainer_e_step(PyBobLearnMiscISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // Parses input arguments in a single shot
  char** kwlist = e_step.kwlist(0);

  PyBobLearnMiscISVBaseObject* isv_base = 0;
  PyObject* stats = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnMiscISVBase_Type, &isv_base,
                                                                 &PyList_Type, &stats)) Py_RETURN_NONE;

  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  if(extract_GMMStats_2d(stats ,training_data)==0)
    self->cxx->eStep(*isv_base->cxx, training_data);

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
.add_prototype("isv_base,stats")
.add_parameter("isv_base", ":py:class:`bob.learn.misc.ISVBase`", "ISVBase Object");
static PyObject* PyBobLearnMiscISVTrainer_m_step(PyBobLearnMiscISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // Parses input arguments in a single shot 
  char** kwlist = m_step.kwlist(0);

  PyBobLearnMiscISVBaseObject* isv_base = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscISVBase_Type, &isv_base)) Py_RETURN_NONE;

  self->cxx->mStep(*isv_base->cxx);

  BOB_CATCH_MEMBER("cannot perform the m_step method", 0)

  Py_RETURN_NONE;
}



/*** enrol ***/
static auto enrol = bob::extension::FunctionDoc(
  "enrol",
  "",
  "",
  true
)
.add_prototype("isv_machine,features,n_iter","")
.add_parameter("isv_machine", ":py:class:`bob.learn.misc.ISVMachine`", "ISVMachine Object")
.add_parameter("features", "list(:py:class:`bob.learn.misc.GMMStats`)`", "")
.add_parameter("n_iter", "int", "Number of iterations");
static PyObject* PyBobLearnMiscISVTrainer_enrol(PyBobLearnMiscISVTrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // Parses input arguments in a single shot
  char** kwlist = enrol.kwlist(0);

  PyBobLearnMiscISVMachineObject* isv_machine = 0;
  PyObject* stats = 0;
  int n_iter = 1;


  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!i", kwlist, &PyBobLearnMiscISVMachine_Type, &isv_machine,
                                                                  &PyList_Type, &stats, &n_iter)) Py_RETURN_NONE;

  std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > training_data;
  if(extract_GMMStats_1d(stats ,training_data)==0)
    self->cxx->enrol(*isv_machine->cxx, training_data, n_iter);

  BOB_CATCH_MEMBER("cannot perform the enrol method", 0)

  Py_RETURN_NONE;
}



static PyMethodDef PyBobLearnMiscISVTrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnMiscISVTrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    e_step.name(),
    (PyCFunction)PyBobLearnMiscISVTrainer_e_step,
    METH_VARARGS|METH_KEYWORDS,
    e_step.doc()
  },
  {
    m_step.name(),
    (PyCFunction)PyBobLearnMiscISVTrainer_m_step,
    METH_VARARGS|METH_KEYWORDS,
    m_step.doc()
  },
  {
    enrol.name(),
    (PyCFunction)PyBobLearnMiscISVTrainer_enrol,
    METH_VARARGS|METH_KEYWORDS,
    enrol.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscISVTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscISVTrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscISVTrainer_Type.tp_name      = ISVTrainer_doc.name();
  PyBobLearnMiscISVTrainer_Type.tp_basicsize = sizeof(PyBobLearnMiscISVTrainerObject);
  PyBobLearnMiscISVTrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance;
  PyBobLearnMiscISVTrainer_Type.tp_doc       = ISVTrainer_doc.doc();

  // set the functions
  PyBobLearnMiscISVTrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnMiscISVTrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnMiscISVTrainer_init);
  PyBobLearnMiscISVTrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnMiscISVTrainer_delete);
  PyBobLearnMiscISVTrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscISVTrainer_RichCompare);
  PyBobLearnMiscISVTrainer_Type.tp_methods      = PyBobLearnMiscISVTrainer_methods;
  PyBobLearnMiscISVTrainer_Type.tp_getset       = PyBobLearnMiscISVTrainer_getseters;
  //PyBobLearnMiscISVTrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnMiscISVTrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscISVTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscISVTrainer_Type);
  return PyModule_AddObject(module, "_ISVTrainer", (PyObject*)&PyBobLearnMiscISVTrainer_Type) >= 0;
}


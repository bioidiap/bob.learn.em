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

//Defining maps for each initializatio method
static const std::map<std::string, bob::learn::em::PLDATrainer::InitFMethod> FMethod = {{"RANDOM_F",  bob::learn::em::PLDATrainer::RANDOM_F}, {"BETWEEN_SCATTER", bob::learn::em::PLDATrainer::BETWEEN_SCATTER}};

static const std::map<std::string, bob::learn::em::PLDATrainer::InitGMethod> GMethod = {{"RANDOM_G",  bob::learn::em::PLDATrainer::RANDOM_G}, {"WITHIN_SCATTER", bob::learn::em::PLDATrainer::WITHIN_SCATTER}};

static const std::map<std::string, bob::learn::em::PLDATrainer::InitSigmaMethod> SigmaMethod = {{"RANDOM_SIGMA",  bob::learn::em::PLDATrainer::RANDOM_SIGMA}, {"VARIANCE_G", bob::learn::em::PLDATrainer::VARIANCE_G}, {"CONSTANT", bob::learn::em::PLDATrainer::CONSTANT}, {"VARIANCE_DATA", bob::learn::em::PLDATrainer::VARIANCE_DATA}};



//String to type
static inline bob::learn::em::PLDATrainer::InitFMethod string2FMethod(const std::string& o){
  auto it = FMethod.find(o);
  if (it == FMethod.end()) throw std::runtime_error("The given FMethod '" + o + "' is not known; choose one of ('RANDOM_F','BETWEEN_SCATTER')");
  else return it->second;
}

static inline bob::learn::em::PLDATrainer::InitGMethod string2GMethod(const std::string& o){
  auto it = GMethod.find(o);
  if (it == GMethod.end()) throw std::runtime_error("The given GMethod '" + o + "' is not known; choose one of ('RANDOM_G','WITHIN_SCATTER')");
  else return it->second;
}

static inline bob::learn::em::PLDATrainer::InitSigmaMethod string2SigmaMethod(const std::string& o){
  auto it = SigmaMethod.find(o);
  if (it == SigmaMethod.end()) throw std::runtime_error("The given SigmaMethod '" + o + "' is not known; choose one of ('RANDOM_SIGMA','VARIANCE_G', 'CONSTANT', 'VARIANCE_DATA')");
  else return it->second;
}

//Type to string
static inline const std::string& FMethod2string(bob::learn::em::PLDATrainer::InitFMethod o){
  for (auto it = FMethod.begin(); it != FMethod.end(); ++it) if (it->second == o) return it->first;
  throw std::runtime_error("The given FMethod type is not known");
}

static inline const std::string& GMethod2string(bob::learn::em::PLDATrainer::InitGMethod o){
  for (auto it = GMethod.begin(); it != GMethod.end(); ++it) if (it->second == o) return it->first;
  throw std::runtime_error("The given GMethod type is not known");
}

static inline const std::string& SigmaMethod2string(bob::learn::em::PLDATrainer::InitSigmaMethod o){
  for (auto it = SigmaMethod.begin(); it != SigmaMethod.end(); ++it) if (it->second == o) return it->first;
  throw std::runtime_error("The given SigmaMethod type is not known");
}


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


/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/


static auto PLDATrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".PLDATrainer",
  "This class can be used to train the :math:`F`, :math:`G` and "
  " :math:`\\Sigma` matrices and the mean vector :math:`\\mu` of a PLDA model."
  "References: [ElShafey2014]_,[PrinceElder2007]_,[LiFu2012]_",
  ""
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

  .add_parameter("other", ":py:class:`bob.learn.em.PLDATrainer`", "A PLDATrainer object to be copied.")
  .add_parameter("use_sum_second_order", "bool", "")
);

static int PyBobLearnEMPLDATrainer_init_copy(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDATrainer_doc.kwlist(1);
  PyBobLearnEMPLDATrainerObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMPLDATrainer_Type, &o)){
    PLDATrainer_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::PLDATrainer(*o->cxx));
  return 0;
}


static int PyBobLearnEMPLDATrainer_init_bool(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDATrainer_doc.kwlist(0);
  PyObject* use_sum_second_order = Py_False;

  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O!", kwlist, &PyBool_Type, &use_sum_second_order))
    return -1;

  self->cxx.reset(new bob::learn::em::PLDATrainer(f(use_sum_second_order)));
  return 0;
}


static int PyBobLearnEMPLDATrainer_init(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if(nargs==0)
    return PyBobLearnEMPLDATrainer_init_bool(self, args, kwargs);
  else if(nargs==1){
    //Reading the input argument
    PyObject* arg = 0;
    if (PyTuple_Size(args))
      arg = PyTuple_GET_ITEM(args, 0);
    else {
      PyObject* tmp = PyDict_Values(kwargs);
      auto tmp_ = make_safe(tmp);
      arg = PyList_GET_ITEM(tmp, 0);
    }

    if(PyBobLearnEMPLDATrainer_Check(arg))
      // If the constructor input is PLDATrainer object
      return PyBobLearnEMPLDATrainer_init_copy(self, args, kwargs);
    else
      return PyBobLearnEMPLDATrainer_init_bool(self, args, kwargs);
  }
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires only 0 or 1 argument, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
    PLDATrainer_doc.print_usage();
    return -1;
  }

  BOB_CATCH_MEMBER("cannot create PLDATrainer", -1)
  return 0;
}


static void PyBobLearnEMPLDATrainer_delete(PyBobLearnEMPLDATrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}


int PyBobLearnEMPLDATrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMPLDATrainer_Type));
}


static PyObject* PyBobLearnEMPLDATrainer_RichCompare(PyBobLearnEMPLDATrainerObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMPLDATrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMPLDATrainerObject*>(other);
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
PyObject* PyBobLearnEMPLDATrainer_get_z_second_order(PyBobLearnEMPLDATrainerObject* self, void*){
  BOB_TRY
  //return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZSecondOrder());
  return vector_as_list(self->cxx->getZSecondOrder());
  BOB_CATCH_MEMBER("z_second_order could not be read", 0)
}


static auto z_second_order_sum = bob::extension::VariableDoc(
  "z_second_order_sum",
  "array_like <float, 2D>",
  "",
  ""
);
PyObject* PyBobLearnEMPLDATrainer_get_z_second_order_sum(PyBobLearnEMPLDATrainerObject* self, void*){
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
PyObject* PyBobLearnEMPLDATrainer_get_z_first_order(PyBobLearnEMPLDATrainerObject* self, void*){
  BOB_TRY
  //return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZFirstOrder());
  return vector_as_list(self->cxx->getZFirstOrder());
  BOB_CATCH_MEMBER("z_first_order could not be read", 0)
}


/***** init_f_method *****/
static auto init_f_method = bob::extension::VariableDoc(
  "init_f_method",
  "str",
  "The method used for the initialization of :math:`$F$`.",
  "Possible values are: ('RANDOM_F', 'BETWEEN_SCATTER')"
);
PyObject* PyBobLearnEMPLDATrainer_getFMethod(PyBobLearnEMPLDATrainerObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("s", FMethod2string(self->cxx->getInitFMethod()).c_str());
  BOB_CATCH_MEMBER("init_f_method method could not be read", 0)
}
int PyBobLearnEMPLDATrainer_setFMethod(PyBobLearnEMPLDATrainerObject* self, PyObject* value, void*) {
  BOB_TRY

  if (!PyString_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an str", Py_TYPE(self)->tp_name, init_f_method.name());
    return -1;
  }
  self->cxx->setInitFMethod(string2FMethod(PyString_AS_STRING(value)));

  return 0;
  BOB_CATCH_MEMBER("init_f_method method could not be set", -1)
}


/***** init_g_method *****/
static auto init_g_method = bob::extension::VariableDoc(
  "init_g_method",
  "str",
  "The method used for the initialization of :math:`$G$`.",
  "Possible values are: ('RANDOM_G', 'WITHIN_SCATTER')"
);
PyObject* PyBobLearnEMPLDATrainer_getGMethod(PyBobLearnEMPLDATrainerObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("s", GMethod2string(self->cxx->getInitGMethod()).c_str());
  BOB_CATCH_MEMBER("init_g_method method could not be read", 0)
}
int PyBobLearnEMPLDATrainer_setGMethod(PyBobLearnEMPLDATrainerObject* self, PyObject* value, void*) {
  BOB_TRY

  if (!PyString_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an str", Py_TYPE(self)->tp_name, init_g_method.name());
    return -1;
  }
  self->cxx->setInitGMethod(string2GMethod(PyString_AS_STRING(value)));

  return 0;
  BOB_CATCH_MEMBER("init_g_method method could not be set", -1)
}

/***** init_sigma_method *****/
static auto init_sigma_method = bob::extension::VariableDoc(
  "init_sigma_method",
  "str",
  "The method used for the initialization of :math:`$\\Sigma$`.",
  "Possible values are: ('RANDOM_SIGMA', 'VARIANCE_G', 'CONSTANT', 'VARIANCE_DATA')"
);
PyObject* PyBobLearnEMPLDATrainer_getSigmaMethod(PyBobLearnEMPLDATrainerObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("s", SigmaMethod2string(self->cxx->getInitSigmaMethod()).c_str());
  BOB_CATCH_MEMBER("init_sigma_method method could not be read", 0)
}
int PyBobLearnEMPLDATrainer_setSigmaMethod(PyBobLearnEMPLDATrainerObject* self, PyObject* value, void*) {
  BOB_TRY

  if (!PyString_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an str", Py_TYPE(self)->tp_name, init_sigma_method.name());
    return -1;
  }
  self->cxx->setInitSigmaMethod(string2SigmaMethod(PyString_AS_STRING(value)));

  return 0;
  BOB_CATCH_MEMBER("init_sigma_method method could not be set", -1)
}


static auto use_sum_second_order = bob::extension::VariableDoc(
  "use_sum_second_order",
  "bool",
  "Tells whether the second order statistics are stored during the training procedure, or only their sum.",
  ""
);
PyObject* PyBobLearnEMPLDATrainer_getUseSumSecondOrder(PyBobLearnEMPLDATrainerObject* self, void*){
  BOB_TRY
  return Py_BuildValue("O",self->cxx->getUseSumSecondOrder()?Py_True:Py_False);
  BOB_CATCH_MEMBER("use_sum_second_order could not be read", 0)
}
int PyBobLearnEMPLDATrainer_setUseSumSecondOrder(PyBobLearnEMPLDATrainerObject* self, PyObject* value, void*) {
  BOB_TRY

  if (!PyBool_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an str", Py_TYPE(self)->tp_name, use_sum_second_order.name());
    return -1;
  }
  self->cxx->setUseSumSecondOrder(f(value));

  return 0;
  BOB_CATCH_MEMBER("use_sum_second_order method could not be set", -1)
}



static PyGetSetDef PyBobLearnEMPLDATrainer_getseters[] = {
  {
   z_first_order.name(),
   (getter)PyBobLearnEMPLDATrainer_get_z_first_order,
   0,
   z_first_order.doc(),
   0
  },
  {
   z_second_order_sum.name(),
   (getter)PyBobLearnEMPLDATrainer_get_z_second_order_sum,
   0,
   z_second_order_sum.doc(),
   0
  },
  {
   z_second_order.name(),
   (getter)PyBobLearnEMPLDATrainer_get_z_second_order,
   0,
   z_second_order.doc(),
   0
  },
  {
   init_f_method.name(),
   (getter)PyBobLearnEMPLDATrainer_getFMethod,
   (setter)PyBobLearnEMPLDATrainer_setFMethod,
   init_f_method.doc(),
   0
  },
  {
   init_g_method.name(),
   (getter)PyBobLearnEMPLDATrainer_getGMethod,
   (setter)PyBobLearnEMPLDATrainer_setGMethod,
   init_g_method.doc(),
   0
  },
  {
   init_sigma_method.name(),
   (getter)PyBobLearnEMPLDATrainer_getSigmaMethod,
   (setter)PyBobLearnEMPLDATrainer_setSigmaMethod,
   init_sigma_method.doc(),
   0
  },
  {
   use_sum_second_order.name(),
   (getter)PyBobLearnEMPLDATrainer_getUseSumSecondOrder,
   (setter)PyBobLearnEMPLDATrainer_setUseSumSecondOrder,
   use_sum_second_order.doc(),
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
.add_prototype("plda_base, data, [rng]")
.add_parameter("plda_base", ":py:class:`bob.learn.em.PLDABase`", "PLDAMachine Object")
.add_parameter("data", "list", "")
.add_parameter("rng", ":py:class:`bob.core.random.mt19937`", "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.");
static PyObject* PyBobLearnEMPLDATrainer_initialize(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = initialize.kwlist(0);

  PyBobLearnEMPLDABaseObject* plda_base = 0;
  PyObject* data = 0;
  PyBoostMt19937Object* rng = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|O!", kwlist, &PyBobLearnEMPLDABase_Type, &plda_base,
                                                                 &PyList_Type, &data,
                                                                 &PyBoostMt19937_Type, &rng)) return 0;

  std::vector<blitz::Array<double,2> > data_vector;
  if(list_as_vector(data ,data_vector)==0){
    if(rng){
      self->cxx->setRng(rng->rng);
    }

    self->cxx->initialize(*plda_base->cxx, data_vector);
  }
  else
    return 0;

  BOB_CATCH_MEMBER("cannot perform the initialize method", 0)

  Py_RETURN_NONE;
}


/*** e_step ***/
static auto e_step = bob::extension::FunctionDoc(
  "e_step",
  "Expectation step before the EM steps",
  "",
  true
)
.add_prototype("plda_base,data")
.add_parameter("plda_base", ":py:class:`bob.learn.em.PLDABase`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnEMPLDATrainer_e_step(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = e_step.kwlist(0);

  PyBobLearnEMPLDABaseObject* plda_base = 0;
  PyObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnEMPLDABase_Type, &plda_base,
                                                                 &PyList_Type, &data)) return 0;

  std::vector<blitz::Array<double,2> > data_vector;
  if(list_as_vector(data ,data_vector)==0)
    self->cxx->eStep(*plda_base->cxx, data_vector);
  else
    return 0;

  BOB_CATCH_MEMBER("cannot perform the e_step method", 0)

  Py_RETURN_NONE;
}


/*** m_step ***/
static auto m_step = bob::extension::FunctionDoc(
  "m_step",
  "Maximization step ",
  "",
  true
)
.add_prototype("plda_base,data")
.add_parameter("plda_base", ":py:class:`bob.learn.em.PLDABase`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnEMPLDATrainer_m_step(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = m_step.kwlist(0);

  PyBobLearnEMPLDABaseObject* plda_base = 0;
  PyObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnEMPLDABase_Type, &plda_base,
                                                                 &PyList_Type, &data)) return 0;

  std::vector<blitz::Array<double,2> > data_vector;
  if(list_as_vector(data ,data_vector)==0)
    self->cxx->mStep(*plda_base->cxx, data_vector);
  else
    return 0;

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
.add_parameter("plda_base", ":py:class:`bob.learn.em.PLDABase`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnEMPLDATrainer_finalize(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = finalize.kwlist(0);

  PyBobLearnEMPLDABaseObject* plda_base = 0;
  PyObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyBobLearnEMPLDABase_Type, &plda_base,
                                                                 &PyList_Type, &data)) return 0;

  std::vector<blitz::Array<double,2> > data_vector;
  if(list_as_vector(data ,data_vector)==0)
    self->cxx->finalize(*plda_base->cxx, data_vector);
  else
    return 0;

  BOB_CATCH_MEMBER("cannot perform the finalize method", 0)

  Py_RETURN_NONE;
}



/*** enroll ***/
static auto enroll = bob::extension::FunctionDoc(
  "enroll",
  "Main procedure for enrolling a PLDAMachine",
  "",
  true
)
.add_prototype("plda_machine,data")
.add_parameter("plda_machine", ":py:class:`bob.learn.em.PLDAMachine`", "PLDAMachine Object")
.add_parameter("data", "list", "");
static PyObject* PyBobLearnEMPLDATrainer_enroll(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = enroll.kwlist(0);

  PyBobLearnEMPLDAMachineObject* plda_machine = 0;
  PyBlitzArrayObject* data = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnEMPLDAMachine_Type, &plda_machine,
                                                                 &PyBlitzArray_Converter, &data)) return 0;

  auto data_ = make_safe(data);
  self->cxx->enroll(*plda_machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(data));

  BOB_CATCH_MEMBER("cannot perform the enroll method", 0)

  Py_RETURN_NONE;
}


/*** is_similar_to ***/
static auto is_similar_to = bob::extension::FunctionDoc(
  "is_similar_to",

  "Compares this PLDATrainer with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.PLDAMachine`", "A PLDAMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMPLDATrainer_IsSimilarTo(PyBobLearnEMPLDATrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMPLDATrainerObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMPLDATrainer_Type, &other,
        &r_epsilon, &a_epsilon)){

        is_similar_to.print_usage();
        return 0;
  }

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}



static PyMethodDef PyBobLearnEMPLDATrainer_methods[] = {
  {
    initialize.name(),
    (PyCFunction)PyBobLearnEMPLDATrainer_initialize,
    METH_VARARGS|METH_KEYWORDS,
    initialize.doc()
  },
  {
    e_step.name(),
    (PyCFunction)PyBobLearnEMPLDATrainer_e_step,
    METH_VARARGS|METH_KEYWORDS,
    e_step.doc()
  },
  {
    m_step.name(),
    (PyCFunction)PyBobLearnEMPLDATrainer_m_step,
    METH_VARARGS|METH_KEYWORDS,
    m_step.doc()
  },
  {
    finalize.name(),
    (PyCFunction)PyBobLearnEMPLDATrainer_finalize,
    METH_VARARGS|METH_KEYWORDS,
    finalize.doc()
  },
  {
    enroll.name(),
    (PyCFunction)PyBobLearnEMPLDATrainer_enroll,
    METH_VARARGS|METH_KEYWORDS,
    enroll.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMPLDATrainer_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMPLDATrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMPLDATrainer(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMPLDATrainer_Type.tp_name      = PLDATrainer_doc.name();
  PyBobLearnEMPLDATrainer_Type.tp_basicsize = sizeof(PyBobLearnEMPLDATrainerObject);
  PyBobLearnEMPLDATrainer_Type.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;//Enable the class inheritance;
  PyBobLearnEMPLDATrainer_Type.tp_doc       = PLDATrainer_doc.doc();

  // set the functions
  PyBobLearnEMPLDATrainer_Type.tp_new          = PyType_GenericNew;
  PyBobLearnEMPLDATrainer_Type.tp_init         = reinterpret_cast<initproc>(PyBobLearnEMPLDATrainer_init);
  PyBobLearnEMPLDATrainer_Type.tp_dealloc      = reinterpret_cast<destructor>(PyBobLearnEMPLDATrainer_delete);
  PyBobLearnEMPLDATrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMPLDATrainer_RichCompare);
  PyBobLearnEMPLDATrainer_Type.tp_methods      = PyBobLearnEMPLDATrainer_methods;
  PyBobLearnEMPLDATrainer_Type.tp_getset       = PyBobLearnEMPLDATrainer_getseters;
  //PyBobLearnEMPLDATrainer_Type.tp_call         = reinterpret_cast<ternaryfunc>(PyBobLearnEMPLDATrainer_compute_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMPLDATrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMPLDATrainer_Type);
  return PyModule_AddObject(module, "PLDATrainer", (PyObject*)&PyBobLearnEMPLDATrainer_Type) >= 0;
}

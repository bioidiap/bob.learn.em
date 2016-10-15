/**
 * @date Thu Jan 30 11:10:15 2015 +0200
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
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

static auto PLDAMachine_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".PLDAMachine",

  "This class is a container for an enrolled identity/class. It contains information extracted from the enrollment samples. "
  "It should be used in combination with a PLDABase instance.\n\n"
  "References: [ElShafey2014]_, [PrinceElder2007]_, [LiFu2012]_",
  ""
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",

     "Constructor, builds a new PLDAMachine.",

    "",
    true
  )
  .add_prototype("plda_base","")
  .add_prototype("other","")
  .add_prototype("hdf5,plda_base","")

  .add_parameter("plda_base", ":py:class:`bob.learn.em.PLDABase`", "")
  .add_parameter("other", ":py:class:`bob.learn.em.PLDAMachine`", "A PLDAMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnEMPLDAMachine_init_copy(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDAMachine_doc.kwlist(1);
  PyBobLearnEMPLDAMachineObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMPLDAMachine_Type, &o)){
    PLDAMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::PLDAMachine(*o->cxx));
  return 0;
}


static int PyBobLearnEMPLDAMachine_init_hdf5(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDAMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  PyBobLearnEMPLDABaseObject* plda_base;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O!", kwlist, &PyBobIoHDF5File_Converter, &config,
                                                                 &PyBobLearnEMPLDABase_Type, &plda_base)){
    PLDAMachine_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);
  self->cxx.reset(new bob::learn::em::PLDAMachine(*(config->f),plda_base->cxx));

  return 0;
}


static int PyBobLearnEMPLDAMachine_init_pldabase(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDAMachine_doc.kwlist(0);
  PyBobLearnEMPLDABaseObject* plda_base;

  //Here we have to select which keyword argument to read
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMPLDABase_Type, &plda_base)){
    PLDAMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::PLDAMachine(plda_base->cxx));
  return 0;
}

static int PyBobLearnEMPLDAMachine_init(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if(nargs==1){
    //Reading the input argument
    PyObject* arg = 0;
    if (PyTuple_Size(args))
      arg = PyTuple_GET_ITEM(args, 0);
    else {
      PyObject* tmp = PyDict_Values(kwargs);
      auto tmp_ = make_safe(tmp);
      arg = PyList_GET_ITEM(tmp, 0);
    }

    // If the constructor input is Gaussian object
    if (PyBobLearnEMPLDAMachine_Check(arg))
      return PyBobLearnEMPLDAMachine_init_copy(self, args, kwargs);
    // If the constructor input is a HDF5
    else if (PyBobLearnEMPLDABase_Check(arg))
      return PyBobLearnEMPLDAMachine_init_pldabase(self, args, kwargs);
  }
  else if(nargs==2)
    return PyBobLearnEMPLDAMachine_init_hdf5(self, args, kwargs);
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
    PLDAMachine_doc.print_usage();
    return -1;
  }
  BOB_CATCH_MEMBER("cannot create PLDAMachine", -1)
  return 0;
}



static void PyBobLearnEMPLDAMachine_delete(PyBobLearnEMPLDAMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMPLDAMachine_RichCompare(PyBobLearnEMPLDAMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMPLDAMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMPLDAMachineObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare PLDAMachine objects", 0)
}

int PyBobLearnEMPLDAMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMPLDAMachine_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int, int)",
  "A tuple that represents the dimensionality of the feature vector dim_d, the :math:`F` matrix and the :math:`G` matrix.",
  ""
);
PyObject* PyBobLearnEMPLDAMachine_getShape(PyBobLearnEMPLDAMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i,i)", self->cxx->getDimD(), self->cxx->getDimF(), self->cxx->getDimG());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}


/***** n_samples *****/
static auto n_samples = bob::extension::VariableDoc(
  "n_samples",
  "int",
  "Number of enrolled samples",
  ""
);
static PyObject* PyBobLearnEMPLDAMachine_getNSamples(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return Py_BuildValue("i",self->cxx->getNSamples());
  BOB_CATCH_MEMBER("n_samples could not be read", 0)
}
int PyBobLearnEMPLDAMachine_setNSamples(PyBobLearnEMPLDAMachineObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, n_samples.name());
    return -1;
  }

  if (PyInt_AS_LONG(value) < 0){
    PyErr_Format(PyExc_TypeError, "n_samples must be greater than or equal to zero");
    return -1;
  }

  self->cxx->setNSamples(PyInt_AS_LONG(value));
  BOB_CATCH_MEMBER("n_samples could not be set", -1)
  return 0;
}


/***** w_sum_xit_beta_xi *****/
static auto w_sum_xit_beta_xi = bob::extension::VariableDoc(
  "w_sum_xit_beta_xi",
  "float",
  "Gets the :math:`A = -0.5 \\sum_{i} x_{i}^T \\beta x_{i}` value",
  ""
);
static PyObject* PyBobLearnEMPLDAMachine_getWSumXitBetaXi(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getWSumXitBetaXi());
  BOB_CATCH_MEMBER("w_sum_xit_beta_xi could not be read", 0)
}
int PyBobLearnEMPLDAMachine_setWSumXitBetaXi(PyBobLearnEMPLDAMachineObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBob_NumberCheck(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an float", Py_TYPE(self)->tp_name, w_sum_xit_beta_xi.name());
    return -1;
  }

  self->cxx->setWSumXitBetaXi(PyFloat_AS_DOUBLE(value));
  BOB_CATCH_MEMBER("w_sum_xit_beta_xi could not be set", -1)
  return 0;
}


/***** plda_base *****/
static auto plda_base = bob::extension::VariableDoc(
  "plda_base",
  ":py:class:`bob.learn.em.PLDABase`",
  "The PLDABase attached to this machine",
  ""
);
PyObject* PyBobLearnEMPLDAMachine_getPLDABase(PyBobLearnEMPLDAMachineObject* self, void*){
  BOB_TRY

  boost::shared_ptr<bob::learn::em::PLDABase> plda_base_o = self->cxx->getPLDABase();

  //Allocating the correspondent python object
  PyBobLearnEMPLDABaseObject* retval =
    (PyBobLearnEMPLDABaseObject*)PyBobLearnEMPLDABase_Type.tp_alloc(&PyBobLearnEMPLDABase_Type, 0);
  retval->cxx = plda_base_o;

  return Py_BuildValue("N",retval);
  BOB_CATCH_MEMBER("plda_base could not be read", 0)
}
int PyBobLearnEMPLDAMachine_setPLDABase(PyBobLearnEMPLDAMachineObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBobLearnEMPLDABase_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a :py:class:`bob.learn.em.PLDABase`", Py_TYPE(self)->tp_name, plda_base.name());
    return -1;
  }

  PyBobLearnEMPLDABaseObject* plda_base_o = 0;
  PyArg_Parse(value, "O!", &PyBobLearnEMPLDABase_Type,&plda_base_o);

  self->cxx->setPLDABase(plda_base_o->cxx);

  return 0;
  BOB_CATCH_MEMBER("plda_base could not be set", -1)
}


/***** weighted_sum *****/
static auto weighted_sum = bob::extension::VariableDoc(
  "weighted_sum",
  "array_like <float, 1D>",
  "Get/Set :math:`\\sum_{i} F^T \\beta x_{i}` value",
  ""
);
static PyObject* PyBobLearnEMPLDAMachine_getWeightedSum(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getWeightedSum());
  BOB_CATCH_MEMBER("weighted_sum could not be read", 0)
}
int PyBobLearnEMPLDAMachine_setWeightedSum(PyBobLearnEMPLDAMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, weighted_sum.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "weighted_sum");
  if (!b) return -1;
  self->cxx->setWeightedSum(*b);
  return 0;
  BOB_CATCH_MEMBER("`weighted_sum` vector could not be set", -1)
}


/***** log_likelihood *****/
static auto log_likelihood = bob::extension::VariableDoc(
  "log_likelihood",
  "float",
  "Get the current log likelihood",
  ""
);
static PyObject* PyBobLearnEMPLDAMachine_getLogLikelihood(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getLogLikelihood());
  BOB_CATCH_MEMBER("log_likelihood could not be read", 0)
}
int PyBobLearnEMPLDAMachine_setLogLikelihood(PyBobLearnEMPLDAMachineObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBob_NumberCheck(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a float", Py_TYPE(self)->tp_name, log_likelihood.name());
    return -1;
  }

  self->cxx->setLogLikelihood(PyFloat_AS_DOUBLE(value));
  BOB_CATCH_MEMBER("log_likelihood could not be set", -1)
  return 0;
}


static PyGetSetDef PyBobLearnEMPLDAMachine_getseters[] = {
  {
   shape.name(),
   (getter)PyBobLearnEMPLDAMachine_getShape,
   0,
   shape.doc(),
   0
  },
  {
   n_samples.name(),
   (getter)PyBobLearnEMPLDAMachine_getNSamples,
   (setter)PyBobLearnEMPLDAMachine_setNSamples,
   n_samples.doc(),
   0
  },
  {
   w_sum_xit_beta_xi.name(),
   (getter)PyBobLearnEMPLDAMachine_getWSumXitBetaXi,
   (setter)PyBobLearnEMPLDAMachine_setWSumXitBetaXi,
   w_sum_xit_beta_xi.doc(),
   0
  },
  {
   plda_base.name(),
   (getter)PyBobLearnEMPLDAMachine_getPLDABase,
   (setter)PyBobLearnEMPLDAMachine_setPLDABase,
   plda_base.doc(),
   0
  },
  {
   weighted_sum.name(),
   (getter)PyBobLearnEMPLDAMachine_getWeightedSum,
   (setter)PyBobLearnEMPLDAMachine_setWeightedSum,
   weighted_sum.doc(),
   0
  },
  {
   log_likelihood.name(),
   (getter)PyBobLearnEMPLDAMachine_getLogLikelihood,
   (setter)PyBobLearnEMPLDAMachine_setLogLikelihood,
   log_likelihood.doc(),
   0
  },
  {0}  // Sentinel
};



/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/


/*** save ***/
static auto save = bob::extension::FunctionDoc(
  "save",
  "Save the configuration of the PLDAMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnEMPLDAMachine_Save(PyBobLearnEMPLDAMachineObject* self,  PyObject* args, PyObject* kwargs) {

  BOB_TRY

  // get list of arguments
  char** kwlist = save.kwlist(0);
  PyBobIoHDF5FileObject* hdf5;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, PyBobIoHDF5File_Converter, &hdf5)) return 0;

  auto hdf5_ = make_safe(hdf5);
  self->cxx->save(*hdf5->f);

  BOB_CATCH_MEMBER("cannot save the data", 0)
  Py_RETURN_NONE;
}

/*** load ***/
static auto load = bob::extension::FunctionDoc(
  "load",
  "Load the configuration of the PLDAMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnEMPLDAMachine_Load(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = load.kwlist(0);
  PyBobIoHDF5FileObject* hdf5;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, PyBobIoHDF5File_Converter, &hdf5)) return 0;

  auto hdf5_ = make_safe(hdf5);
  self->cxx->load(*hdf5->f);

  BOB_CATCH_MEMBER("cannot load the data", 0)
  Py_RETURN_NONE;
}


/*** is_similar_to ***/
static auto is_similar_to = bob::extension::FunctionDoc(
  "is_similar_to",

  "Compares this PLDAMachine with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.PLDAMachine`", "A PLDAMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMPLDAMachine_IsSimilarTo(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMPLDAMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMPLDAMachine_Type, &other,
        &r_epsilon, &a_epsilon)){

        is_similar_to.print_usage();
        return 0;
  }

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}


/***** get_gamma *****/
static auto get_gamma = bob::extension::FunctionDoc(
  "get_gamma",
  "Gets the :math:`\\gamma_a` matrix for a given :math:`a` (number of samples). "
  ":math:`\\gamma_{a}=(Id + a F^T \\beta F)^{-1}= \\mathcal{F}_{a}`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","array_like <float, 2D>","Get the :math:`\\gamma` matrix");
static PyObject* PyBobLearnEMPLDAMachine_getGamma(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_gamma.kwlist(0);

  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) return 0;

  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getGamma(i));
  BOB_CATCH_MEMBER("`get_gamma` could not be read", 0)
}


/***** has_gamma *****/
static auto has_gamma = bob::extension::FunctionDoc(
  "has_gamma",
  "Tells if the :math:`\\gamma_a` matrix for a given a (number of samples) exists. "
  ":math:`\\gamma_a=(Id + a F^T \\beta F)^{-1}`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","bool","");
static PyObject* PyBobLearnEMPLDAMachine_hasGamma(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = has_gamma.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) return 0;

  if(self->cxx->hasGamma(i))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
 BOB_CATCH_MEMBER("`has_gamma` could not be read", 0)
}


/***** get_add_gamma *****/
static auto get_add_gamma = bob::extension::FunctionDoc(
  "get_add_gamma",
   "Gets the :math:`gamma_a` matrix for a given :math:`f_a` (number of samples)."
   " :math:`\\gamma_a=(Id + a F^T \\beta F)^{-1} =\\mathcal{F}_{a}`."
   "Tries to find it from the base machine and then from this machine.",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","array_like <float, 2D>","");
static PyObject* PyBobLearnEMPLDAMachine_getAddGamma(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_add_gamma.kwlist(0);

  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) return 0;

  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getAddGamma(i));
  BOB_CATCH_MEMBER("`get_add_gamma` could not be read", 0)
}


/***** has_log_like_const_term *****/
static auto has_log_like_const_term = bob::extension::FunctionDoc(
  "has_log_like_const_term",
   "Tells if the log likelihood constant term for a given :math:`a` (number of samples) exists in this machine (does not check the base machine). "
   ":math:`l_{a}=\\frac{a}{2} ( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| +log|\\gamma_a|)`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","bool","");
static PyObject* PyBobLearnEMPLDAMachine_hasLogLikeConstTerm(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = has_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) return 0;

  if(self->cxx->hasLogLikeConstTerm(i))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
 BOB_CATCH_MEMBER("`has_log_like_const_term` could not be read", 0)
}


/***** get_add_log_like_const_term *****/
static auto get_add_log_like_const_term = bob::extension::FunctionDoc(
  "get_add_log_like_const_term",

   "Gets the log likelihood constant term for a given :math:`a` (number of samples). "
   ":math:`l_{a} = \\frac{a}{2} ( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| +log|gamma_a|)`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","float","");
static PyObject* PyBobLearnEMPLDAMachine_getAddLogLikeConstTerm(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_add_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) return 0;

  return Py_BuildValue("d",self->cxx->getAddLogLikeConstTerm(i));

  BOB_CATCH_MEMBER("`get_add_log_like_const_term` could not be read", 0)
}


/***** get_log_like_const_term *****/
static auto get_log_like_const_term = bob::extension::FunctionDoc(
  "get_log_like_const_term",
   "Gets the log likelihood constant term for a given :math:`a` (number of samples). "
    ":math:`l_{a}=\\frac{a}{2}( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| + log|\\gamma_a|)`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","float","");
static PyObject* PyBobLearnEMPLDAMachine_getLogLikeConstTerm(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) return 0;

  return Py_BuildValue("d",self->cxx->getLogLikeConstTerm(i));

  BOB_CATCH_MEMBER("`get_log_like_const_term` could not be read", 0)
}

/***** clear_maps *****/
static auto clear_maps = bob::extension::FunctionDoc(
  "clear_maps",
  "Clears the maps (:math:`\\gamma_a` and loglike_constterm_a).",
  0,
  true
)
.add_prototype("");
static PyObject* PyBobLearnEMPLDAMachine_clearMaps(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  self->cxx->clearMaps();
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("`clear_maps` could not be read", 0)
}


/***** compute_log_likelihood *****/
static auto compute_log_likelihood = bob::extension::FunctionDoc(
  "compute_log_likelihood",
  "Compute the log-likelihood of the given sample and (optionally) the enrolled samples",
  0,
  true
)
.add_prototype("sample,with_enrolled_samples","output")
.add_parameter("sample", "array_like <float, 1D>,array_like <float, 2D>", "Sample")
.add_parameter("with_enrolled_samples", "bool", "")
.add_return("output","float","The log-likelihood");
static PyObject* PyBobLearnEMPLDAMachine_computeLogLikelihood(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = compute_log_likelihood.kwlist(0);

  PyBlitzArrayObject* samples;
  PyObject* with_enrolled_samples = Py_True;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|O!", kwlist, &PyBlitzArray_Converter, &samples,
                                                                  &PyBool_Type, &with_enrolled_samples)) return 0;
  auto samples_ = make_safe(samples);

  /*Using the proper method according to the dimension*/
  if (samples->ndim==1)
    return Py_BuildValue("d",self->cxx->computeLogLikelihood(*PyBlitzArrayCxx_AsBlitz<double,1>(samples), f(with_enrolled_samples)));
  else
    return Py_BuildValue("d",self->cxx->computeLogLikelihood(*PyBlitzArrayCxx_AsBlitz<double,2>(samples), f(with_enrolled_samples)));


  BOB_CATCH_MEMBER("`compute_log_likelihood` could not be read", 0)
}


/***** log_likelihood_ratio *****/
static auto log_likelihood_ratio = bob::extension::FunctionDoc(
  "log_likelihood_ratio",
  "Computes a log likelihood ratio from a 1D or 2D blitz::Array",
  0,
  true
)
.add_prototype("samples","output")
.add_parameter("samples", "array_like <float, 1D>,array_like <float, 2D>", "Sample")
.add_return("output","float","The log-likelihood ratio");
static PyObject* PyBobLearnEMPLDAMachine_log_likelihood_ratio(PyBobLearnEMPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = log_likelihood_ratio.kwlist(0);

  PyBlitzArrayObject* samples;

  /*Convert to PyObject first to access the number of dimensions*/
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &samples)) return 0;
  auto samples_ = make_safe(samples);

   //There are 2 methods in C++, one <double,1> and the another <double,2>
  if(samples->ndim==1)
    return Py_BuildValue("d",self->cxx->forward(*PyBlitzArrayCxx_AsBlitz<double,1>(samples)));
  else
    return Py_BuildValue("d",self->cxx->forward(*PyBlitzArrayCxx_AsBlitz<double,2>(samples)));

  BOB_CATCH_MEMBER("log_likelihood_ratio could not be executed", 0)
}


static PyMethodDef PyBobLearnEMPLDAMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    get_gamma.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_getGamma,
    METH_VARARGS|METH_KEYWORDS,
    get_gamma.doc()
  },
  {
    has_gamma.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_hasGamma,
    METH_VARARGS|METH_KEYWORDS,
    has_gamma.doc()
  },
  {
    get_add_gamma.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_getAddGamma,
    METH_VARARGS|METH_KEYWORDS,
    get_add_gamma.doc()
  },
  {
    has_log_like_const_term.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_hasLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    has_log_like_const_term.doc()
  },
  {
    get_add_log_like_const_term.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_getAddLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    get_add_log_like_const_term.doc()
  },
  {
    get_log_like_const_term.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_getLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    get_log_like_const_term.doc()
  },
  {
    clear_maps.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_clearMaps,
    METH_NOARGS,
    clear_maps.doc()
  },
  {
    compute_log_likelihood.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_computeLogLikelihood,
    METH_VARARGS|METH_KEYWORDS,
    compute_log_likelihood.doc()
  },
  {
    log_likelihood_ratio.name(),
    (PyCFunction)PyBobLearnEMPLDAMachine_log_likelihood_ratio,
    METH_VARARGS|METH_KEYWORDS,
    log_likelihood_ratio.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the JFA type struct; will be initialized later
PyTypeObject PyBobLearnEMPLDAMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMPLDAMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMPLDAMachine_Type.tp_name      = PLDAMachine_doc.name();
  PyBobLearnEMPLDAMachine_Type.tp_basicsize = sizeof(PyBobLearnEMPLDAMachineObject);
  PyBobLearnEMPLDAMachine_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMPLDAMachine_Type.tp_doc       = PLDAMachine_doc.doc();

  // set the functions
  PyBobLearnEMPLDAMachine_Type.tp_new         = PyType_GenericNew;
  PyBobLearnEMPLDAMachine_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnEMPLDAMachine_init);
  PyBobLearnEMPLDAMachine_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnEMPLDAMachine_delete);
  PyBobLearnEMPLDAMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMPLDAMachine_RichCompare);
  PyBobLearnEMPLDAMachine_Type.tp_methods     = PyBobLearnEMPLDAMachine_methods;
  PyBobLearnEMPLDAMachine_Type.tp_getset      = PyBobLearnEMPLDAMachine_getseters;
  PyBobLearnEMPLDAMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMPLDAMachine_log_likelihood_ratio);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMPLDAMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMPLDAMachine_Type);
  return PyModule_AddObject(module, "PLDAMachine", (PyObject*)&PyBobLearnEMPLDAMachine_Type) >= 0;
}

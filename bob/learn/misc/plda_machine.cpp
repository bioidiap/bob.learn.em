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

static auto PLDAMachine_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".PLDAMachine",

  "This class is a container for an enrolled identity/class. It contains information extracted from the enrollment samples. "
  "It should be used in combination with a PLDABase instance."
  "References: [ElShafey2014,PrinceElder2007,LiFu2012]",
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

  .add_parameter("plda_base", "`bob.learn.misc.PLDABase`", "")  
  .add_parameter("other", ":py:class:`bob.learn.misc.PLDAMachine`", "A PLDAMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnMiscPLDAMachine_init_copy(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDAMachine_doc.kwlist(1);
  PyBobLearnMiscPLDAMachineObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscPLDAMachine_Type, &o)){
    PLDAMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::PLDAMachine(*o->cxx));
  return 0;
}


static int PyBobLearnMiscPLDAMachine_init_hdf5(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDAMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  PyBobLearnMiscPLDABaseObject* plda_base;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O!", kwlist, &PyBobIoHDF5File_Converter, &config,
                                                                 &PyBobLearnMiscPLDABase_Type, &plda_base)){
    PLDAMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::PLDAMachine(*(config->f),plda_base->cxx));

  return 0;
}


static int PyBobLearnMiscPLDAMachine_init_pldabase(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDAMachine_doc.kwlist(0);  
  PyBobLearnMiscPLDABaseObject* plda_base;
  
  //Here we have to select which keyword argument to read  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscPLDABase_Type, &plda_base)){
    PLDAMachine_doc.print_usage();
    return -1;
  }
  
  self->cxx.reset(new bob::learn::misc::PLDAMachine(plda_base->cxx));
  return 0;
}

static int PyBobLearnMiscPLDAMachine_init(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
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
    if (PyBobLearnMiscPLDAMachine_Check(arg))
      return PyBobLearnMiscPLDAMachine_init_copy(self, args, kwargs);
    // If the constructor input is a HDF5
    else if (PyBobLearnMiscPLDABase_Check(arg))
      return PyBobLearnMiscPLDAMachine_init_pldabase(self, args, kwargs);
  }
  else if(nargs==2)
    return PyBobLearnMiscPLDAMachine_init_hdf5(self, args, kwargs);
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
    PLDAMachine_doc.print_usage();
    return -1;
  }
  BOB_CATCH_MEMBER("cannot create PLDAMachine", 0)
  return 0;
}



static void PyBobLearnMiscPLDAMachine_delete(PyBobLearnMiscPLDAMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnMiscPLDAMachine_RichCompare(PyBobLearnMiscPLDAMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscPLDAMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscPLDAMachineObject*>(other);
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

int PyBobLearnMiscPLDAMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscPLDAMachine_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int, int)",
  "A tuple that represents the dimensionality of the feature vector :math:`dim_d`, the :math:`F` matrix and the :math:`G` matrix.",
  ""
);
PyObject* PyBobLearnMiscPLDAMachine_getShape(PyBobLearnMiscPLDAMachineObject* self, void*) {
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
static PyObject* PyBobLearnMiscPLDAMachine_getNSamples(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getNSamples());
  BOB_CATCH_MEMBER("n_samples could not be read", 0)
}
int PyBobLearnMiscPLDAMachine_setNSamples(PyBobLearnMiscPLDAMachineObject* self, PyObject* value, void*){
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
  "double",
  "Gets the :math:`A = -0.5 \\sum_{i} x_{i}^T \\beta x_{i}` value",
  ""
);
static PyObject* PyBobLearnMiscPLDAMachine_getWSumXitBetaXi(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getWSumXitBetaXi());
  BOB_CATCH_MEMBER("w_sum_xit_beta_xi could not be read", 0)
}
int PyBobLearnMiscPLDAMachine_setWSumXitBetaXi(PyBobLearnMiscPLDAMachineObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyNumber_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an float", Py_TYPE(self)->tp_name, w_sum_xit_beta_xi.name());
    return -1;
  }

  self->cxx->setWSumXitBetaXi(PyFloat_AS_DOUBLE(value));
  BOB_CATCH_MEMBER("w_sum_xit_beta_xi could not be set", -1)
  return 0;
}

static PyGetSetDef PyBobLearnMiscPLDAMachine_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnMiscPLDAMachine_getShape,
   0,
   shape.doc(),
   0
  },  
  {
   n_samples.name(),
   (getter)PyBobLearnMiscPLDAMachine_getNSamples,
   (setter)PyBobLearnMiscPLDAMachine_setNSamples,
   n_samples.doc(),
   0
  },  
  {
   w_sum_xit_beta_xi.name(),
   (getter)PyBobLearnMiscPLDAMachine_getWSumXitBetaXi,
   (setter)PyBobLearnMiscPLDAMachine_setWSumXitBetaXi,
   w_sum_xit_beta_xi.doc(),
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
static PyObject* PyBobLearnMiscPLDAMachine_Save(PyBobLearnMiscPLDAMachineObject* self,  PyObject* args, PyObject* kwargs) {

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
static PyObject* PyBobLearnMiscPLDAMachine_Load(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
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
.add_parameter("other", ":py:class:`bob.learn.misc.PLDAMachine`", "A PLDAMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnMiscPLDAMachine_IsSimilarTo(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnMiscPLDAMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnMiscPLDAMachine_Type, &other,
        &r_epsilon, &a_epsilon)){

        is_similar_to.print_usage(); 
        return 0;        
  }

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}


/***** gamma *****/
static auto gamma_var = bob::extension::FunctionDoc(
  "gamma",
  "Gets the :math:`\\gamma_a` matrix for a given :math:`a` (number of samples). "
  ":math:`gamma_{a} = (Id + a F^T \beta F)^{-1} = \\mathcal{F}_{a}`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","array_like <float, 2D>","Get the :math:`\\gamma` matrix");
static PyObject* PyBobLearnMiscPLDAMachine_getGamma(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = gamma_var.kwlist(0);

  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getGamma(i));
  BOB_CATCH_MEMBER("`gamma` could not be read", 0)
}


/***** has_gamma *****/
static auto has_gamma = bob::extension::FunctionDoc(
  "has_gamma",
  "Tells if the :math:`gamma_a` matrix for a given a (number of samples) exists. "
  ":math:`gamma_a=(Id + a F^T \\beta F)^{-1}`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","bool","");
static PyObject* PyBobLearnMiscPLDAMachine_hasGamma(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = has_gamma.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

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
   ":math:`gamma_a = (Id + a F^T \\beta F)^{-1} = \\mathcal{F}_{a}`."
   "Tries to find it from the base machine and then from this machine.",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","array_like <float, 2D>","");
static PyObject* PyBobLearnMiscPLDAMachine_getAddGamma(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_add_gamma.kwlist(0);

  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

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
static PyObject* PyBobLearnMiscPLDAMachine_hasLogLikeConstTerm(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = has_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

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
   ":math:`l_{a} = \\frac{a}{2} ( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| +log|\\gamma_a|)`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","double","");
static PyObject* PyBobLearnMiscPLDAMachine_getAddLogLikeConstTerm(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_add_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  return Py_BuildValue("d",self->cxx->getAddLogLikeConstTerm(i));

  BOB_CATCH_MEMBER("`get_add_log_like_const_term` could not be read", 0)    
}


/***** get_log_like_const_term *****/
static auto get_log_like_const_term = bob::extension::FunctionDoc(
  "get_log_like_const_term",
   "Gets the log likelihood constant term for a given :math:`a` (number of samples). "
    ":math:`l_{a} = \\frac{a}{2} ( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| +log|\\gamma_a|)",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","double","");
static PyObject* PyBobLearnMiscPLDAMachine_getLogLikeConstTerm(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  return Py_BuildValue("d",self->cxx->getLogLikeConstTerm(i));

  BOB_CATCH_MEMBER("`get_log_like_const_term` could not be read", 0)    
}

/***** clear_maps *****/
static auto clear_maps = bob::extension::FunctionDoc(
  "clear_maps",
  "Clears the maps (:math:`gamma_a` and loglike_constterm_a).",
  0,
  true
);
static PyObject* PyBobLearnMiscPLDAMachine_clearMaps(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  self->cxx->clearMaps();
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("`clear_maps` could not be read", 0)    
}


/***** compute_log_likelihood *****/
/*
static auto compute_log_likelihood = bob::extension::FunctionDoc(
  "compute_log_likelihood",
  "Compute the log-likelihood of the given sample and (optionally) the enrolled samples",
  0,
  true
)
.add_prototype("sample,use_enrolled_samples","output")
.add_parameter("sample", "array_like <float, 1D>", "Sample")
.add_parameter("use_enrolled_samples", "bool", "")
.add_return("output","double","The log-likelihood");
static PyObject* PyBobLearnMiscPLDAMachine_computeLogLikelihood(PyBobLearnMiscPLDAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = compute_log_likelihood.kwlist(0);
  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O!", kwlist, )) Py_RETURN_NONE;

  return Py_BuildValue("d",self->cxx->getLogLikeConstTerm(i));

  BOB_CATCH_MEMBER("`get_log_like_const_term` could not be read", 0)    
}
*/

static PyMethodDef PyBobLearnMiscPLDAMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    gamma_var.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_getGamma,
    METH_VARARGS|METH_KEYWORDS,
    gamma_var.doc()
  },
  {
    has_gamma.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_hasGamma,
    METH_VARARGS|METH_KEYWORDS,
    has_gamma.doc()
  },
  {
    get_add_gamma.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_getAddGamma,
    METH_VARARGS|METH_KEYWORDS,
    get_add_gamma.doc()
  },
  {
    has_log_like_const_term.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_hasLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    has_log_like_const_term.doc()
  },  
  {
    get_add_log_like_const_term.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_getAddLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    get_add_log_like_const_term.doc()
  },  
  {
    get_log_like_const_term.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_getLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    get_log_like_const_term.doc()
  },  
  {
    clear_maps.name(),
    (PyCFunction)PyBobLearnMiscPLDAMachine_clearMaps,
    METH_NOARGS,
    clear_maps.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the JFA type struct; will be initialized later
PyTypeObject PyBobLearnMiscPLDAMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscPLDAMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscPLDAMachine_Type.tp_name      = PLDAMachine_doc.name();
  PyBobLearnMiscPLDAMachine_Type.tp_basicsize = sizeof(PyBobLearnMiscPLDAMachineObject);
  PyBobLearnMiscPLDAMachine_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscPLDAMachine_Type.tp_doc       = PLDAMachine_doc.doc();

  // set the functions
  PyBobLearnMiscPLDAMachine_Type.tp_new         = PyType_GenericNew;
  PyBobLearnMiscPLDAMachine_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnMiscPLDAMachine_init);
  PyBobLearnMiscPLDAMachine_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnMiscPLDAMachine_delete);
  PyBobLearnMiscPLDAMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscPLDAMachine_RichCompare);
  PyBobLearnMiscPLDAMachine_Type.tp_methods     = PyBobLearnMiscPLDAMachine_methods;
  PyBobLearnMiscPLDAMachine_Type.tp_getset      = PyBobLearnMiscPLDAMachine_getseters;
  //PyBobLearnMiscPLDAMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnMiscPLDAMachine_forward);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscPLDAMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscPLDAMachine_Type);
  return PyModule_AddObject(module, "PLDAMachine", (PyObject*)&PyBobLearnMiscPLDAMachine_Type) >= 0;
}


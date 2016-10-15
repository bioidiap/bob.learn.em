/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Fri 21 Nov 10:38:48 2013
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto Gaussian_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".Gaussian",
  "This class implements a multivariate diagonal Gaussian distribution"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new multivariate gaussian object",
    "",
    true
  )
  .add_prototype("n_inputs","")
  .add_prototype("other","")
  .add_prototype("hdf5","")
  .add_prototype("","")

  .add_parameter("n_inputs", "int", "Dimension of the feature vector")
  .add_parameter("other", ":py:class:`bob.learn.em.GMMStats`", "A GMMStats object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")
);



static int PyBobLearnEMGaussian_init_number(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = Gaussian_doc.kwlist(0);
  int n_inputs=1;
  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &n_inputs))
    return -1;

  if(n_inputs < 0){
    PyErr_Format(PyExc_TypeError, "input argument must be greater than or equal to zero");
    Gaussian_doc.print_usage();
    return -1;
   }

  self->cxx.reset(new bob::learn::em::Gaussian(n_inputs));
  return 0;
}

static int PyBobLearnEMGaussian_init_copy(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = Gaussian_doc.kwlist(1);
  PyBobLearnEMGaussianObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMGaussian_Type, &tt)){
    Gaussian_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::Gaussian(*tt->cxx));
  return 0;
}

static int PyBobLearnEMGaussian_init_hdf5(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = Gaussian_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    Gaussian_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);
  
  self->cxx.reset(new bob::learn::em::Gaussian(*(config->f)));

  return 0;
}


static int PyBobLearnEMGaussian_init(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {

  BOB_TRY

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);
  if (nargs==0){
    self->cxx.reset(new bob::learn::em::Gaussian());
    return 0;
  }

  //Reading the input argument
  PyObject* arg = 0;
  if (PyTuple_Size(args)) 
    arg = PyTuple_GET_ITEM(args, 0);
  else {
    PyObject* tmp = PyDict_Values(kwargs);
    auto tmp_ = make_safe(tmp);
    arg = PyList_GET_ITEM(tmp, 0);
  }

  /**If the constructor input is a number**/
  if (PyBob_NumberCheck(arg)) 
    return PyBobLearnEMGaussian_init_number(self, args, kwargs);
  /**If the constructor input is Gaussian object**/
  else if (PyBobLearnEMGaussian_Check(arg))
    return PyBobLearnEMGaussian_init_copy(self, args, kwargs);
  /**If the constructor input is a HDF5**/
  else if (PyBobIoHDF5File_Check(arg))
    return PyBobLearnEMGaussian_init_hdf5(self, args, kwargs);
  else
    PyErr_Format(PyExc_TypeError, "invalid input argument");
    Gaussian_doc.print_usage();
    return -1;

  BOB_CATCH_MEMBER("cannot create Gaussian", -1)
  return 0;
}



static void PyBobLearnEMGaussian_delete(PyBobLearnEMGaussianObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMGaussian_RichCompare(PyBobLearnEMGaussianObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMGaussian_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMGaussianObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare Gaussian objects", 0)
}

int PyBobLearnEMGaussian_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMGaussian_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** MEAN *****/
static auto mean = bob::extension::VariableDoc(
  "mean",
  "array_like <float, 1D>",
  "Mean of the Gaussian",
  ""
);
PyObject* PyBobLearnEMGaussian_getMean(PyBobLearnEMGaussianObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getMean());
  BOB_CATCH_MEMBER("mean could not be read", 0)
}
int PyBobLearnEMGaussian_setMean(PyBobLearnEMGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, mean.name());
    return -1;
  }
  
  // perform check on the input  
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, mean.name());
    return -1;
  }  

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, mean.name());
    return -1;
  }  

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0], mean.name());
    return -1;
  }  

  auto o_ = make_safe(input);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "mean");
  if (!b) return -1;
  self->cxx->setMean(*b);
  return 0;
  BOB_CATCH_MEMBER("mean could not be set", -1)
}

/***** Variance *****/
static auto variance = bob::extension::VariableDoc(
  "variance",
  "array_like <float, 1D>",
  "Variance of the Gaussian",
  ""
);
PyObject* PyBobLearnEMGaussian_getVariance(PyBobLearnEMGaussianObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getVariance());
  BOB_CATCH_MEMBER("variance could not be read", 0)
}
int PyBobLearnEMGaussian_setVariance(PyBobLearnEMGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, variance.name());
    return -1;
  }
  auto input_ = make_safe(input);
  
  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, variance.name());
    return -1;
  }  

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, variance.name());
    return -1;
  }  

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0], variance.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "variance");
  if (!b) return -1;
  self->cxx->setVariance(*b);
  return 0;
  BOB_CATCH_MEMBER("variance could not be set", -1)
}


/***** variance_thresholds *****/
static auto variance_thresholds = bob::extension::VariableDoc(
  "variance_thresholds",
  "array_like <float, 1D>",
  "The variance flooring thresholds, i.e. the minimum allowed value of variance in each dimension. ",
  "The variance will be set to this value if an attempt is made to set it to a smaller value."
);
PyObject* PyBobLearnEMGaussian_getVarianceThresholds(PyBobLearnEMGaussianObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getVarianceThresholds());
  BOB_CATCH_MEMBER("variance_thresholds could not be read", 0)
}
int PyBobLearnEMGaussian_setVarianceThresholds(PyBobLearnEMGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, variance_thresholds.name());
    return -1;
  }      

  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, variance_thresholds.name());
    return -1;
  }  

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, variance_thresholds.name());
    return -1;
  }  

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0], variance_thresholds.name());
    return -1;
  }
  
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "variance_thresholds");
  if (!b) return -1;
  self->cxx->setVarianceThresholds(*b);
  return 0;
  BOB_CATCH_MEMBER("variance_thresholds could not be set", -1)  
}


/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int)",
  "A tuple that represents the dimensionality of the Gaussian ``(dim,)``.",
  ""
);
PyObject* PyBobLearnEMGaussian_getShape(PyBobLearnEMGaussianObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i)", self->cxx->getNInputs());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}

static PyGetSetDef PyBobLearnEMGaussian_getseters[] = {
    {
      mean.name(),
      (getter)PyBobLearnEMGaussian_getMean,
      (setter)PyBobLearnEMGaussian_setMean,
      mean.doc(),
      0
    },
    {
      variance.name(),
      (getter)PyBobLearnEMGaussian_getVariance,
      (setter)PyBobLearnEMGaussian_setVariance,
      variance.doc(),
     0
     },
     {
      variance_thresholds.name(),
      (getter)PyBobLearnEMGaussian_getVarianceThresholds,
      (setter)PyBobLearnEMGaussian_setVarianceThresholds,
      variance_thresholds.doc(),
      0
     },
     {
      shape.name(),
      (getter)PyBobLearnEMGaussian_getShape,
      0,
      shape.doc(),
      0
     },

    {0}  // Sentinel
};


/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

/*** resize ***/
static auto resize = bob::extension::FunctionDoc(
  "resize",
  "Set the input dimensionality, reset the mean to zero and the variance to one."
)
.add_prototype("input")
.add_parameter("input", "int", "Dimensionality of the feature vector");
static PyObject* PyBobLearnEMGaussian_resize(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  int input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &input)) return 0;
  if (input <= 0){
    PyErr_Format(PyExc_TypeError, "input must be greater than zero");
    resize.print_usage();
    return 0;
  }
  self->cxx->setNInputs(input);

  BOB_CATCH_MEMBER("cannot perform the resize method", 0)

  Py_RETURN_NONE;
}

/*** log_likelihood ***/
static auto log_likelihood = bob::extension::FunctionDoc(
  "log_likelihood",
  "Output the log likelihood of the sample, x. The input size is checked.",
  ".. note:: The ``__call__`` function is an alias for this.",
  true
)
.add_prototype("input","output")
.add_parameter("input", "array_like <float, 1D>", "Input vector")
.add_return("output","float","The log likelihood");
static PyObject* PyBobLearnEMGaussian_loglikelihood(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = log_likelihood.kwlist(0);

  PyBlitzArrayObject* input = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) return 0;
  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `input`", Py_TYPE(self)->tp_name);
    log_likelihood.print_usage();
    return 0;
  }  

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64", Py_TYPE(self)->tp_name);
    log_likelihood.print_usage();
    return 0;
  }  

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0]);
    log_likelihood.print_usage();
    return 0;
  }  

  double value = self->cxx->logLikelihood(*PyBlitzArrayCxx_AsBlitz<double,1>(input));
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot compute the likelihood", 0)
}


/*** log_likelihood_ ***/
static auto log_likelihood_ = bob::extension::FunctionDoc(
  "log_likelihood_",
  "Output the log likelihood given a sample. The input size is NOT checked."
)
.add_prototype("input","output")
.add_parameter("input", "array_like <float, 1D>", "Input vector")
.add_return("output","float","The log likelihood");
static PyObject* PyBobLearnEMGaussian_loglikelihood_(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = log_likelihood_.kwlist(0);

  PyBlitzArrayObject* input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) return 0;
  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `input`", Py_TYPE(self)->tp_name);
    log_likelihood.print_usage();
    return 0;
  }  

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64", Py_TYPE(self)->tp_name);
    log_likelihood.print_usage();
    return 0;
  }  

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0]);
    log_likelihood.print_usage();
    return 0;
  }  

  double value = self->cxx->logLikelihood_(*PyBlitzArrayCxx_AsBlitz<double,1>(input));
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot compute the likelihood", 0)
}


/*** save ***/
static auto save = bob::extension::FunctionDoc(
  "save",
  "Save the configuration of the Gassian Machine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnEMGaussian_Save(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {
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
  "Load the configuration of the Gassian Machine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnEMGaussian_Load(PyBobLearnEMGaussianObject* self,  PyObject* args, PyObject* kwargs) {

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
  
  "Compares this Gaussian with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` and any other values internal to this machine.",
  true
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.Gaussian`", "A gaussian to be compared.")
.add_parameter("[r_epsilon]", "float", "Relative precision.")
.add_parameter("[a_epsilon]", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMGaussian_IsSimilarTo(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  PyBobLearnEMGaussianObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMGaussian_Type, &other,
        &r_epsilon, &a_epsilon)) return 0;

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}


/*** set_variance_thresholds ***/
static auto set_variance_thresholds = bob::extension::FunctionDoc(
  "set_variance_thresholds",
  "Set the variance flooring thresholds equal to the given threshold for all the dimensions."
)
.add_prototype("input")
.add_parameter("input","float","Threshold")
;
static PyObject* PyBobLearnEMGaussian_SetVarianceThresholds(PyBobLearnEMGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = set_variance_thresholds.kwlist(0);

  double input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &input)) return 0;

  self->cxx->setVarianceThresholds(input);

  BOB_CATCH_MEMBER("cannot perform the set_variance_Thresholds method", 0)

  Py_RETURN_NONE;
}


static PyMethodDef PyBobLearnEMGaussian_methods[] = {
  {
    resize.name(),
    (PyCFunction)PyBobLearnEMGaussian_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  {
    log_likelihood.name(),
    (PyCFunction)PyBobLearnEMGaussian_loglikelihood,
    METH_VARARGS|METH_KEYWORDS,
    log_likelihood.doc()
  },
  {
    log_likelihood_.name(),
    (PyCFunction)PyBobLearnEMGaussian_loglikelihood_,
    METH_VARARGS|METH_KEYWORDS,
    log_likelihood_.doc()
  },
  {
    save.name(),
    (PyCFunction)PyBobLearnEMGaussian_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMGaussian_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMGaussian_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    set_variance_thresholds.name(),
    (PyCFunction)PyBobLearnEMGaussian_SetVarianceThresholds,
    METH_VARARGS|METH_KEYWORDS,
    set_variance_thresholds.doc()
  },

  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMGaussian_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMGaussian(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMGaussian_Type.tp_name = Gaussian_doc.name();
  PyBobLearnEMGaussian_Type.tp_basicsize = sizeof(PyBobLearnEMGaussianObject);
  PyBobLearnEMGaussian_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMGaussian_Type.tp_doc = Gaussian_doc.doc();

  // set the functions
  PyBobLearnEMGaussian_Type.tp_new = PyType_GenericNew;
  PyBobLearnEMGaussian_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnEMGaussian_init);
  PyBobLearnEMGaussian_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnEMGaussian_delete);
  PyBobLearnEMGaussian_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMGaussian_RichCompare);
  PyBobLearnEMGaussian_Type.tp_methods = PyBobLearnEMGaussian_methods;
  PyBobLearnEMGaussian_Type.tp_getset = PyBobLearnEMGaussian_getseters;
  PyBobLearnEMGaussian_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMGaussian_loglikelihood);

  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMGaussian_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMGaussian_Type);
  return PyModule_AddObject(module, "Gaussian", (PyObject*)&PyBobLearnEMGaussian_Type) >= 0;
}


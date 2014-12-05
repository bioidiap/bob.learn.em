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
  .add_prototype("n_inputs")
  .add_prototype("other")
  .add_prototype("hdf5")
  .add_prototype("")

  .add_parameter("n_inputs", "int", "Dimension of the feature vector")
  .add_parameter("other", ":py:class:`bob.learn.misc.GMMStats`", "A GMMStats object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")
);



static int PyBobLearnMiscGaussian_init_number(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = Gaussian_doc.kwlist(0);
  int n_inputs=1;
  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &n_inputs))
    return -1;

  if(n_inputs < 0){
    PyErr_Format(PyExc_TypeError, "input argument must be greater than or equal to zero");
    return -1;
   }

  self->cxx.reset(new bob::learn::misc::Gaussian(n_inputs));
  return 0;
}

static int PyBobLearnMiscGaussian_init_copy(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = Gaussian_doc.kwlist(1);
  PyBobLearnMiscGaussianObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGaussian_Type, &tt)) return -1;

  self->cxx.reset(new bob::learn::misc::Gaussian(*tt->cxx));
  return 0;
}

static int PyBobLearnMiscGaussian_init_hdf5(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = Gaussian_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)) 
    return -1;

  try {
    self->cxx.reset(new bob::learn::misc::Gaussian(*(config->f)));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;
}


static int PyBobLearnMiscGaussian_init(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {

  BOB_TRY

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);
  if (nargs==0){
    self->cxx.reset(new bob::learn::misc::Gaussian());
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
  if (PyNumber_Check(arg)) 
    return PyBobLearnMiscGaussian_init_number(self, args, kwargs);
  /**If the constructor input is Gaussian object**/
  else if (PyBobLearnMiscGaussian_Check(arg))
    return PyBobLearnMiscGaussian_init_copy(self, args, kwargs);
  /**If the constructor input is a HDF5**/
  else if (PyBobIoHDF5File_Check(arg))
    return PyBobLearnMiscGaussian_init_hdf5(self, args, kwargs);
  else
    PyErr_Format(PyExc_TypeError, "invalid input argument");
    return -1;

  BOB_CATCH_MEMBER("cannot create Gaussian", 0)
  return -0;
}



static void PyBobLearnMiscGaussian_delete(PyBobLearnMiscGaussianObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnMiscGaussian_RichCompare(PyBobLearnMiscGaussianObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscGaussian_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscGaussianObject*>(other);
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

int PyBobLearnMiscGaussian_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscGaussian_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** MEAN *****/
static auto mean = bob::extension::VariableDoc(
  "mean",
  "array_like <double, 1D>"
  "Mean of the Gaussian",
  ""
);
PyObject* PyBobLearnMiscGaussian_getMean(PyBobLearnMiscGaussianObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getMean());
  BOB_CATCH_MEMBER("mean could not be read", 0)
}
int PyBobLearnMiscGaussian_setMean(PyBobLearnMiscGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, mean.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "mean");
  if (!b) return -1;
  self->cxx->setMean(*b);
  return 0;
  BOB_CATCH_MEMBER("mean could not be set", -1)
}

/***** Variance *****/
static auto variance = bob::extension::VariableDoc(
  "variance",
  "array_like <double, 1D>"
  "Variance of the Gaussian",
  ""
);
PyObject* PyBobLearnMiscGaussian_getVariance(PyBobLearnMiscGaussianObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getVariance());
  BOB_CATCH_MEMBER("variance could not be read", 0)
}
int PyBobLearnMiscGaussian_setVariance(PyBobLearnMiscGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, variance.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "variance");
  if (!b) return -1;
  self->cxx->setVariance(*b);
  return 0;
  BOB_CATCH_MEMBER("variance could not be set", -1)
}

/***** dim_d *****/
static auto dimD = bob::extension::VariableDoc(
  "dim_d",
  "int"
  "Dimensionality of the input feature space",
  ""
);
PyObject* PyBobLearnMiscGaussian_getdimD(PyBobLearnMiscGaussianObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getNInputs());
  BOB_CATCH_MEMBER("dimD could not be read", 0)
}
int PyBobLearnMiscGaussian_setdimD(PyBobLearnMiscGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, dimD.name());
    return -1;
  }
  if (PyInt_AS_LONG(value) <= 0){
    PyErr_Format(PyExc_TypeError, "dim_d must be greater than zero");
    return -1;
  }
  self->cxx->setNInputs(PyInt_AS_LONG(value));
  return 0;
  BOB_CATCH_MEMBER("dim_d could not be set", -1)
}


/***** variance_thresholds *****/
static auto variance_thresholds = bob::extension::VariableDoc(
  "variance_thresholds",
  "array_like <double, 1D>"
  "The variance flooring thresholds, i.e. the minimum allowed value of variance in each dimension. ",
  "The variance will be set to this value if an attempt is made to set it to a smaller value."
);
PyObject* PyBobLearnMiscGaussian_getVarianceThresholds(PyBobLearnMiscGaussianObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getVarianceThresholds());
  BOB_CATCH_MEMBER("variance_thresholds could not be read", 0)
}
int PyBobLearnMiscGaussian_setVarianceThresholds(PyBobLearnMiscGaussianObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, variance_thresholds.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "variance_thresholds");
  if (!b) return -1;
  self->cxx->setVarianceThresholds(*b);
  return 0;
  BOB_CATCH_MEMBER("variance_thresholds could not be set", -1)  
}


/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int)"
  "A tuple that represents the dimensionality of the Gaussian ``(dim_d,)``.",
  ""
);
PyObject* PyBobLearnMiscGaussian_getShape(PyBobLearnMiscGaussianObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(n)", self->cxx->getNInputs());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}
int PyBobLearnMiscGaussian_setShape(PyBobLearnMiscGaussianObject* self, PyObject* o, void*){
  BOB_TRY

  if (!PySequence_Check(o)) {
    PyErr_Format(PyExc_TypeError, "`%s' shape can only be set using tuples (or sequences), not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  } 
 
  //getting the shape
  PyObject* shape = PySequence_Tuple(o);
  auto shape_ = make_safe(shape);
  Py_ssize_t dim_d = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape, 0), PyExc_OverflowError);

  self->cxx->setNInputs(dim_d);
  return 0;

  BOB_CATCH_MEMBER("variance_thresholds could not be set", -1)  
}


static PyGetSetDef PyBobLearnMiscGaussian_getseters[] = {
    {
      mean.name(),
      (getter)PyBobLearnMiscGaussian_getMean,
      (setter)PyBobLearnMiscGaussian_setMean,
      mean.doc(),
      0
    },
    {
      variance.name(),
      (getter)PyBobLearnMiscGaussian_getVariance,
      (setter)PyBobLearnMiscGaussian_setVariance,
      variance.doc(),
     0
     },
     {
      dimD.name(),
      (getter)PyBobLearnMiscGaussian_getdimD,
      (setter)PyBobLearnMiscGaussian_setdimD,
      dimD.doc(),
      0
     },
     {
      variance_thresholds.name(),
      (getter)PyBobLearnMiscGaussian_getVarianceThresholds,
      (setter)PyBobLearnMiscGaussian_setVarianceThresholds,
      variance_thresholds.doc(),
      0
     },
     {
      shape.name(),
      (getter)PyBobLearnMiscGaussian_getShape,
      (setter)PyBobLearnMiscGaussian_setShape,
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
  "int"
  "Set the input dimensionality, reset the mean to zero and the variance to one.",
  ""
)
.add_prototype("input")
.add_parameter("input", "int", "Dimensionality of the feature vector");
static PyObject* PyBobLearnMiscGaussian_resize(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  Py_ssize_t input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", kwlist, &input)) Py_RETURN_NONE;
  if (input <= 0){
    PyErr_Format(PyExc_TypeError, "input must be greater than zero");
    Py_RETURN_NONE;
  }
  self->cxx->setNInputs(input);

  BOB_CATCH_MEMBER("cannot perform the resize method", 0)

  Py_RETURN_NONE;
}

/*** log_likelihood ***/
static auto forward = bob::extension::FunctionDoc(
  "forward",
  "array_like <double, 1D> "
  "Output the log likelihood of the sample, x. The input size is checked.",
  ""
)
.add_prototype("input","double")
.add_parameter("input", "array_like <double, 1D>", "Input vector")
.add_return("double","double","The log likelihood");
/*** log_likelihood ***/
static auto log_likelihood = bob::extension::FunctionDoc(
  "log_likelihood",
  "array_like <double, 1D> "
  "Output the log likelihood of the sample, x. The input size is checked."
)
.add_prototype("input","double")
.add_parameter("input", "array_like <double, 1D>", "Input vector")
.add_return("double","double","The log likelihood");
static PyObject* PyBobLearnMiscGaussian_loglikelihood(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = log_likelihood.kwlist(0);

  PyBlitzArrayObject* input = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) Py_RETURN_NONE;
  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  double value = self->cxx->logLikelihood(*PyBlitzArrayCxx_AsBlitz<double,1>(input));
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot compute the likelihood", 0)
}


/*** log_likelihood_ ***/
static auto log_likelihood_ = bob::extension::FunctionDoc(
  "log_likelihood_",
  "array_like <double, 1D> "
  "Output the log likelihood given a sample. The input size is NOT checked.",
  ""
)
.add_prototype("input","double")
.add_parameter("input", "array_like <double, 1D>", "Input vector")
.add_return("double","double","The log likelihood");
static PyObject* PyBobLearnMiscGaussian_loglikelihood_(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = log_likelihood_.kwlist(0);

  PyBlitzArrayObject* input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) Py_RETURN_NONE;
  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  double value = self->cxx->logLikelihood_(*PyBlitzArrayCxx_AsBlitz<double,1>(input));
  return Py_BuildValue("d", value);

  BOB_CATCH_MEMBER("cannot compute the likelihood", 0)
}


/*** save ***/
static auto save = bob::extension::FunctionDoc(
  "save",
  "Save the configuration of the Gassian Machine to a given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing")
;
static PyObject* PyBobLearnMiscGaussian_Save(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {
  // get list of arguments
  char** kwlist = save.kwlist(0);
  PyBobIoHDF5FileObject* hdf5 = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, PyBobIoHDF5File_Converter, &hdf5)){
    save.print_usage();
    return NULL;
  }
  auto hdf5_ = make_safe(hdf5);

  try {
    self->cxx->save(*hdf5->f);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "`%s' cannot write data to file `%s' (at group `%s'): unknown exception caught", Py_TYPE(self)->tp_name,
        hdf5->f->filename().c_str(), hdf5->f->cwd().c_str());
    return 0;
  }

  Py_RETURN_NONE;
}

/*** load ***/
static auto load = bob::extension::FunctionDoc(
  "load",
  "Load the configuration of the Gassian Machine to a given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnMiscGaussian_Load(PyBobLearnMiscGaussianObject* self, PyObject* f) {

  if (!PyBobIoHDF5File_Check(f)) {
    PyErr_Format(PyExc_TypeError, "`%s' cannot load itself from `%s', only from an HDF5 file", Py_TYPE(self)->tp_name, Py_TYPE(f)->tp_name);
    return 0;
  }

  auto h5f = reinterpret_cast<PyBobIoHDF5FileObject*>(f);
  try {
    self->cxx->load(*h5f->f);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot read data from file `%s' (at group `%s'): unknown exception caught", h5f->f->filename().c_str(),
        h5f->f->cwd().c_str());
    return 0;
  }
  Py_RETURN_NONE;
}


/*** is_similar_to ***/
static auto is_similar_to = bob::extension::FunctionDoc(
  "is_similar_to",
  
  "Compares this Gaussian with the ``other`` one to be approximately the same."
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the"
  "relative and absolute precision for the ``weights``, ``biases`` and any other values internal to this machine.",
  0,
  true
)
.add_prototype("other, [r_epsilon], [a_epsilon]","bool")
.add_parameter("other", ":py:class:`bob.learn.misc.Gaussian`", "A gaussian to be compared.")
.add_parameter("[r_epsilon]", "float", "Relative precision.")
.add_parameter("[a_epsilon]", "float", "Absolute precision.")
.add_return("bool","","");
static PyObject* PyBobLearnMiscGaussian_IsSimilarTo(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  PyBobLearnMiscGaussianObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnMiscGaussian_Type, &other,
        &r_epsilon, &a_epsilon)) return 0;

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}


/*** set_variance_thresholds ***/
static auto set_variance_thresholds = bob::extension::FunctionDoc(
  "set_variance_thresholds",
  "int"
  "Set the variance flooring thresholds equal to the given threshold for all the dimensions."
)
.add_prototype("input")
.add_parameter("input","float","Threshold")
;
static PyObject* PyBobLearnMiscGaussian_SetVarianceThresholds(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = set_variance_thresholds.kwlist(0);

  double input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &input)) return 0;

  self->cxx->setVarianceThresholds(input);

  BOB_CATCH_MEMBER("cannot perform the set_variance_Thresholds method", 0)

  Py_RETURN_NONE;
}


static PyMethodDef PyBobLearnMiscGaussian_methods[] = {
  {
    resize.name(),
    (PyCFunction)PyBobLearnMiscGaussian_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  {
    log_likelihood.name(),
    (PyCFunction)PyBobLearnMiscGaussian_loglikelihood,
    METH_VARARGS|METH_KEYWORDS,
    log_likelihood.doc()
  },
  {
    forward.name(),
    (PyCFunction)PyBobLearnMiscGaussian_loglikelihood,
    METH_VARARGS|METH_KEYWORDS,
    forward.doc()
  },
  {
    log_likelihood_.name(),
    (PyCFunction)PyBobLearnMiscGaussian_loglikelihood_,
    METH_VARARGS|METH_KEYWORDS,
    log_likelihood_.doc()
  },
  {
    save.name(),
    (PyCFunction)PyBobLearnMiscGaussian_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnMiscGaussian_Load,
    METH_O,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnMiscGaussian_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    set_variance_thresholds.name(),
    (PyCFunction)PyBobLearnMiscGaussian_SetVarianceThresholds,
    METH_VARARGS|METH_KEYWORDS,
    set_variance_thresholds.doc()
  },

  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscGaussian_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscGaussian(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscGaussian_Type.tp_name = Gaussian_doc.name();
  PyBobLearnMiscGaussian_Type.tp_basicsize = sizeof(PyBobLearnMiscGaussianObject);
  PyBobLearnMiscGaussian_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscGaussian_Type.tp_doc = Gaussian_doc.doc();

  // set the functions
  PyBobLearnMiscGaussian_Type.tp_new = PyType_GenericNew;
  PyBobLearnMiscGaussian_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnMiscGaussian_init);
  PyBobLearnMiscGaussian_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnMiscGaussian_delete);
  PyBobLearnMiscGaussian_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscGaussian_RichCompare);
  PyBobLearnMiscGaussian_Type.tp_methods = PyBobLearnMiscGaussian_methods;
  PyBobLearnMiscGaussian_Type.tp_getset = PyBobLearnMiscGaussian_getseters;
  PyBobLearnMiscGaussian_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnMiscGaussian_loglikelihood);

  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscGaussian_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscGaussian_Type);
  return PyModule_AddObject(module, "Gaussian", (PyObject*)&PyBobLearnMiscGaussian_Type) >= 0;
}


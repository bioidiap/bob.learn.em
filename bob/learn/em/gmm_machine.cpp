/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Wed 11 Dec 18:01:00 2014
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto GMMMachine_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GMMMachine",
  "This class implements a multivariate diagonal Gaussian distribution.",
  "See Section 2.3.9 of Bishop, \"Pattern recognition and machine learning\", 2006"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a GMMMachine",
    "",
    true
  )
  .add_prototype("n_gaussians,n_inputs","")
  .add_prototype("other","")
  .add_prototype("hdf5","")
  .add_prototype("","")

  .add_parameter("n_gaussians", "int", "Number of gaussians")
  .add_parameter("n_inputs", "int", "Dimension of the feature vector")
  .add_parameter("other", ":py:class:`bob.learn.em.GMMMachine`", "A GMMMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnEMGMMMachine_init_number(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMMachine_doc.kwlist(0);
  int n_inputs    = 1;
  int n_gaussians = 1;
  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &n_gaussians, &n_inputs))
    return -1;

  if(n_gaussians < 0){
    PyErr_Format(PyExc_TypeError, "gaussians argument must be greater than or equal to zero");
    return -1;
  }

  if(n_inputs < 0){
    PyErr_Format(PyExc_TypeError, "input argument must be greater than or equal to zero");
    return -1;
   }

  self->cxx.reset(new bob::learn::em::GMMMachine(n_gaussians, n_inputs));
  return 0;
}


static int PyBobLearnEMGMMMachine_init_copy(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMMachine_doc.kwlist(1);
  PyBobLearnEMGMMMachineObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMGMMMachine_Type, &tt)){
    GMMMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::GMMMachine(*tt->cxx));
  return 0;
}


static int PyBobLearnEMGMMMachine_init_hdf5(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    GMMMachine_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);

  self->cxx.reset(new bob::learn::em::GMMMachine(*(config->f)));

  return 0;
}



static int PyBobLearnEMGMMMachine_init(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs) {

    case 0: //default initializer ()
      self->cxx.reset(new bob::learn::em::GMMMachine());
      return 0;

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

      // If the constructor input is Gaussian object
     if (PyBobLearnEMGMMMachine_Check(arg))
       return PyBobLearnEMGMMMachine_init_copy(self, args, kwargs);
      // If the constructor input is a HDF5
     else if (PyBobIoHDF5File_Check(arg))
       return PyBobLearnEMGMMMachine_init_hdf5(self, args, kwargs);
    }
    case 2:
      return PyBobLearnEMGMMMachine_init_number(self, args, kwargs);
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0, 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      GMMMachine_doc.print_usage();
      return -1;
  }
  BOB_CATCH_MEMBER("cannot create GMMMachine", -1)
  return 0;
}



static void PyBobLearnEMGMMMachine_delete(PyBobLearnEMGMMMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMGMMMachine_RichCompare(PyBobLearnEMGMMMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMGMMMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMGMMMachineObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare GMMMachine objects", 0)
}

int PyBobLearnEMGMMMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMGMMMachine_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int)",
  "A tuple that represents the number of gaussians and dimensionality of each Gaussian ``(n_gaussians, dim)``.",
  ""
);
PyObject* PyBobLearnEMGMMMachine_getShape(PyBobLearnEMGMMMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i)", self->cxx->getNGaussians(), self->cxx->getNInputs());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}

/***** MEAN *****/

static auto means = bob::extension::VariableDoc(
  "means",
  "array_like <float, 2D>",
  "The means of the gaussians",
  ""
);
PyObject* PyBobLearnEMGMMMachine_getMeans(PyBobLearnEMGMMMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getMeans());
  BOB_CATCH_MEMBER("means could not be read", 0)
}
int PyBobLearnEMGMMMachine_setMeans(PyBobLearnEMGMMMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, means.name());
    return -1;
  }
  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, means.name());
    return -1;
  }

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, means.name());
    return -1;
  }

  if (input->shape[1] != (Py_ssize_t)self->cxx->getNInputs() && input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNGaussians(), self->cxx->getNInputs(), input->shape[1], input->shape[0], means.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "means");
  if (!b) return -1;
  self->cxx->setMeans(*b);
  return 0;
  BOB_CATCH_MEMBER("means could not be set", -1)
}

/***** Variance *****/
static auto variances = bob::extension::VariableDoc(
  "variances",
  "array_like <float, 2D>",
  "Variances of the gaussians",
  ""
);
PyObject* PyBobLearnEMGMMMachine_getVariances(PyBobLearnEMGMMMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getVariances());
  BOB_CATCH_MEMBER("variances could not be read", 0)
}
int PyBobLearnEMGMMMachine_setVariances(PyBobLearnEMGMMMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, variances.name());
    return -1;
  }
  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, variances.name());
    return -1;
  }

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, variances.name());
    return -1;
  }

  if (input->shape[1] != (Py_ssize_t)self->cxx->getNInputs() && input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNGaussians(), self->cxx->getNInputs(), input->shape[1], input->shape[0], variances.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "variances");
  if (!b) return -1;
  self->cxx->setVariances(*b);
  return 0;
  BOB_CATCH_MEMBER("variances could not be set", -1)
}

/***** Weights *****/
static auto weights = bob::extension::VariableDoc(
  "weights",
  "array_like <float, 1D>",
  "The weights (also known as \"mixing coefficients\")",
  ""
);
PyObject* PyBobLearnEMGMMMachine_getWeights(PyBobLearnEMGMMMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getWeights());
  BOB_CATCH_MEMBER("weights could not be read", 0)
}
int PyBobLearnEMGMMMachine_setWeights(PyBobLearnEMGMMMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, weights.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, weights.name());
    return -1;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, weights.name());
    return -1;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNGaussians(), input->shape[0], weights.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "weights");
  if (!b) return -1;
  self->cxx->setWeights(*b);
  return 0;
  BOB_CATCH_MEMBER("weights could not be set", -1)
}


/***** variance_supervector *****/
static auto variance_supervector = bob::extension::VariableDoc(
  "variance_supervector",
  "array_like <float, 1D>",
  "The variance supervector of the GMMMachine",
  "Concatenation of the variance vectors of each Gaussian of the GMMMachine"
);
PyObject* PyBobLearnEMGMMMachine_getVarianceSupervector(PyBobLearnEMGMMMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getVarianceSupervector());
  BOB_CATCH_MEMBER("variance_supervector could not be read", 0)
}
int PyBobLearnEMGMMMachine_setVarianceSupervector(PyBobLearnEMGMMMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, variance_supervector.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, variance_supervector.name());
    return -1;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, variance_supervector.name());
    return -1;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()*(Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNGaussians()*(Py_ssize_t)self->cxx->getNInputs(), input->shape[0], variance_supervector.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "variance_supervector");
  if (!b) return -1;
  self->cxx->setVarianceSupervector(*b);
  return 0;
  BOB_CATCH_MEMBER("variance_supervector could not be set", -1)
}

/***** mean_supervector *****/
static auto mean_supervector = bob::extension::VariableDoc(
  "mean_supervector",
  "array_like <float, 1D>",
  "The mean supervector of the GMMMachine",
  "Concatenation of the mean vectors of each Gaussian of the GMMMachine"
);
PyObject* PyBobLearnEMGMMMachine_getMeanSupervector(PyBobLearnEMGMMMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getMeanSupervector());
  BOB_CATCH_MEMBER("mean_supervector could not be read", 0)
}
int PyBobLearnEMGMMMachine_setMeanSupervector(PyBobLearnEMGMMMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, mean_supervector.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, mean_supervector.name());
    return -1;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, mean_supervector.name());
    return -1;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()*(Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNGaussians()*(Py_ssize_t)self->cxx->getNInputs(), input->shape[0], mean_supervector.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "mean_supervector");
  if (!b) return -1;
  self->cxx->setMeanSupervector(*b);
  return 0;
  BOB_CATCH_MEMBER("mean_supervector could not be set", -1)
}



/***** variance_thresholds *****/
static auto variance_thresholds = bob::extension::VariableDoc(
  "variance_thresholds",
  "array_like <float, 2D>",
  "Set the variance flooring thresholds in each dimension to the same vector for all Gaussian components if the argument is a 1D numpy arrray, and equal for all Gaussian components and dimensions if the parameter is a scalar. ",
  ""
);
PyObject* PyBobLearnEMGMMMachine_getVarianceThresholds(PyBobLearnEMGMMMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getVarianceThresholds());
  BOB_CATCH_MEMBER("variance_thresholds could not be read", 0)
}
int PyBobLearnEMGMMMachine_setVarianceThresholds(PyBobLearnEMGMMMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, variance_thresholds.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, variance_thresholds.name());
    return -1;
  }

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, variance_thresholds.name());
    return -1;
  }

  if (input->shape[1] != (Py_ssize_t)self->cxx->getNInputs() && input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNGaussians(), self->cxx->getNInputs(), input->shape[1], input->shape[0], variance_thresholds.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "variance_thresholds");
  if (!b) return -1;
  self->cxx->setVarianceThresholds(*b);
  return 0;
  BOB_CATCH_MEMBER("variance_thresholds could not be set", -1)
}




static PyGetSetDef PyBobLearnEMGMMMachine_getseters[] = {
  {
   shape.name(),
   (getter)PyBobLearnEMGMMMachine_getShape,
   0,
   shape.doc(),
   0
  },
  {
   means.name(),
   (getter)PyBobLearnEMGMMMachine_getMeans,
   (setter)PyBobLearnEMGMMMachine_setMeans,
   means.doc(),
   0
  },
  {
   variances.name(),
   (getter)PyBobLearnEMGMMMachine_getVariances,
   (setter)PyBobLearnEMGMMMachine_setVariances,
   variances.doc(),
   0
  },
  {
   weights.name(),
   (getter)PyBobLearnEMGMMMachine_getWeights,
   (setter)PyBobLearnEMGMMMachine_setWeights,
   weights.doc(),
   0
  },
  {
   variance_thresholds.name(),
   (getter)PyBobLearnEMGMMMachine_getVarianceThresholds,
   (setter)PyBobLearnEMGMMMachine_setVarianceThresholds,
   variance_thresholds.doc(),
   0
  },
  {
   variance_supervector.name(),
   (getter)PyBobLearnEMGMMMachine_getVarianceSupervector,
   (setter)PyBobLearnEMGMMMachine_setVarianceSupervector,
   variance_supervector.doc(),
   0
  },

  {
   mean_supervector.name(),
   (getter)PyBobLearnEMGMMMachine_getMeanSupervector,
   (setter)PyBobLearnEMGMMMachine_setMeanSupervector,
   mean_supervector.doc(),
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
  "Save the configuration of the GMMMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnEMGMMMachine_Save(PyBobLearnEMGMMMachineObject* self,  PyObject* args, PyObject* kwargs) {

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
  "Load the configuration of the GMMMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnEMGMMMachine_Load(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
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

  "Compares this GMMMachine with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.GMMMachine`", "A GMMMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMGMMMachine_IsSimilarTo(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMGMMMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMGMMMachine_Type, &other,
        &r_epsilon, &a_epsilon)){

        is_similar_to.print_usage();
        return 0;
  }

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}


/*** resize ***/
static auto resize = bob::extension::FunctionDoc(
  "resize",
  "Allocates space for the statistics and resets to zero.",
  0,
  true
)
.add_prototype("n_gaussians,n_inputs")
.add_parameter("n_gaussians", "int", "Number of gaussians")
.add_parameter("n_inputs", "int", "Dimensionality of the feature vector");
static PyObject* PyBobLearnEMGMMMachine_resize(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  int n_gaussians = 0;
  int n_inputs = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &n_gaussians, &n_inputs)) Py_RETURN_NONE;

  if (n_gaussians <= 0){
    PyErr_Format(PyExc_TypeError, "n_gaussians must be greater than zero");
    resize.print_usage();
    return 0;
  }
  if (n_inputs <= 0){
    PyErr_Format(PyExc_TypeError, "n_inputs must be greater than zero");
    resize.print_usage();
    return 0;
  }

  self->cxx->resize(n_gaussians, n_inputs);

  BOB_CATCH_MEMBER("cannot perform the resize method", 0)

  Py_RETURN_NONE;
}


/*** log_likelihood ***/
static auto log_likelihood = bob::extension::FunctionDoc(
  "log_likelihood",
  "Output the log likelihood of the sample, x, i.e. :math:`log(p(x|GMM))`. Inputs are checked.",
  ".. note:: The ``__call__`` function is an alias for this. \n "
  "If `input` is 2D the average along the samples will be computed (:math:`\\frac{log(p(x|GMM))}{N}`) ",
  true
)
.add_prototype("input","output")
.add_parameter("input", "array_like <float, 1D>", "Input vector")
.add_return("output","float","The log likelihood");
static PyObject* PyBobLearnEMGMMMachine_loglikelihood(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
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

  if (input->ndim > 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D or 2D arrays of float64", Py_TYPE(self)->tp_name);
    log_likelihood.print_usage();
    return 0;
  }

  int shape_index = input->ndim - 1; //Getting the index of the dimensionality (0 for 1D arrays, 1 for 2D arrays)

  if (input->shape[shape_index] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0]);
    log_likelihood.print_usage();
    return 0;
  }

  double value = 0;
  if (input->ndim == 1)
    value = self->cxx->logLikelihood(*PyBlitzArrayCxx_AsBlitz<double,1>(input));
  else
    value = self->cxx->logLikelihood(*PyBlitzArrayCxx_AsBlitz<double,2>(input));


  return Py_BuildValue("d", value);
  BOB_CATCH_MEMBER("cannot compute the likelihood", 0)
}


/*** log_likelihood_ ***/
static auto log_likelihood_ = bob::extension::FunctionDoc(
  "log_likelihood_",
  "Output the log likelihood of the sample, x, i.e. :math:`log(p(x|GMM))`. Inputs are NOT checked.",
  "",
  true
)
.add_prototype("input","output")
.add_parameter("input", "array_like <float, 1D>", "Input vector")
.add_return("output","float","The log likelihood");
static PyObject* PyBobLearnEMGMMMachine_loglikelihood_(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
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


/*** acc_statistics ***/
static auto acc_statistics = bob::extension::FunctionDoc(
  "acc_statistics",
  "Accumulate the GMM statistics for this sample(s). Inputs are checked.",
  "",
  true
)
.add_prototype("input,stats")
.add_parameter("input", "array_like <float, 2D>", "Input vector")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "Statistics of the GMM");
static PyObject* PyBobLearnEMGMMMachine_accStatistics(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = acc_statistics.kwlist(0);

  PyBlitzArrayObject* input           = 0;
  PyBobLearnEMGMMStatsObject* stats = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O!", kwlist, &PyBlitzArray_Converter,&input,
                                                                 &PyBobLearnEMGMMStats_Type, &stats))
    return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  if (input->ndim == 1)
    self->cxx->accStatistics(*PyBlitzArrayCxx_AsBlitz<double,1>(input), *stats->cxx);
  else
    self->cxx->accStatistics(*PyBlitzArrayCxx_AsBlitz<double,2>(input), *stats->cxx);


  BOB_CATCH_MEMBER("cannot accumulate the statistics", 0)
  Py_RETURN_NONE;
}


/*** acc_statistics_ ***/
static auto acc_statistics_ = bob::extension::FunctionDoc(
  "acc_statistics_",
  "Accumulate the GMM statistics for this sample(s). Inputs are NOT checked.",
  "",
  true
)
.add_prototype("input,stats")
.add_parameter("input", "array_like <float, 2D>", "Input vector")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "Statistics of the GMM");
static PyObject* PyBobLearnEMGMMMachine_accStatistics_(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = acc_statistics_.kwlist(0);

  PyBlitzArrayObject* input = 0;
  PyBobLearnEMGMMStatsObject* stats = 0;

  if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O!", kwlist, &PyBlitzArray_Converter,&input,
                                                                &PyBobLearnEMGMMStats_Type, &stats))
    return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  if (input->ndim==1)
    self->cxx->accStatistics_(*PyBlitzArrayCxx_AsBlitz<double,1>(input), *stats->cxx);
  else
    self->cxx->accStatistics_(*PyBlitzArrayCxx_AsBlitz<double,2>(input), *stats->cxx);

  BOB_CATCH_MEMBER("cannot accumulate the statistics", 0)
  Py_RETURN_NONE;
}



/*** set_variance_thresholds ***/
static auto set_variance_thresholds = bob::extension::FunctionDoc(
  "set_variance_thresholds",
  "Set the variance flooring thresholds in each dimension to the same vector for all Gaussian components if the argument is a 1D numpy arrray, and equal for all Gaussian components and dimensions if the parameter is a scalar.",
  "",
  true
)
.add_prototype("input")
.add_parameter("input", "float or array_like <float, 1D>", "The new variance threshold, or a vector of thresholds for all Gaussian components");
static PyObject* PyBobLearnEMGMMMachine_setVarianceThresholds_method(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = set_variance_thresholds.kwlist(0);

  PyBlitzArrayObject* input_array = 0;
  double input_number = 0;
  if(PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &input_number)){
    self->cxx->setVarianceThresholds(input_number);
  }
  else if(PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input_array)) {
    //protects acquired resources through this scope
    auto input_ = make_safe(input_array);
    self->cxx->setVarianceThresholds(*PyBlitzArrayCxx_AsBlitz<double,1>(input_array));
  }
  else
    return 0;

  // clear any error that might have been set in the functions above
  PyErr_Clear();

  BOB_CATCH_MEMBER("cannot accumulate set the variance threshold", 0)
  Py_RETURN_NONE;
}




/*** get_gaussian ***/
static auto get_gaussian = bob::extension::FunctionDoc(
  "get_gaussian",
  "Get the specified Gaussian component.",
  ".. note:: An exception is thrown if i is out of range.",
  true
)
.add_prototype("i","gaussian")
.add_parameter("i", "int", "Index of the gaussian")
.add_return("gaussian",":py:class:`bob.learn.em.Gaussian`","Gaussian object");
static PyObject* PyBobLearnEMGMMMachine_get_gaussian(PyBobLearnEMGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_gaussian.kwlist(0);

  int i = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) return 0;

  //Allocating the correspondent python object
  PyBobLearnEMGaussianObject* retval =
    (PyBobLearnEMGaussianObject*)PyBobLearnEMGaussian_Type.tp_alloc(&PyBobLearnEMGaussian_Type, 0);

  retval->cxx = self->cxx->getGaussian(i);

  return Py_BuildValue("N",retval);

  BOB_CATCH_MEMBER("cannot compute the likelihood", 0)
}



static PyMethodDef PyBobLearnEMGMMMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  {
    log_likelihood.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_loglikelihood,
    METH_VARARGS|METH_KEYWORDS,
    log_likelihood.doc()
  },
  {
    log_likelihood_.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_loglikelihood_,
    METH_VARARGS|METH_KEYWORDS,
    log_likelihood_.doc()
  },
  {
    acc_statistics.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_accStatistics,
    METH_VARARGS|METH_KEYWORDS,
    acc_statistics.doc()
  },
  {
    acc_statistics_.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_accStatistics_,
    METH_VARARGS|METH_KEYWORDS,
    acc_statistics_.doc()
  },

  {
    get_gaussian.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_get_gaussian,
    METH_VARARGS|METH_KEYWORDS,
    get_gaussian.doc()
  },

  {
    set_variance_thresholds.name(),
    (PyCFunction)PyBobLearnEMGMMMachine_setVarianceThresholds_method,
    METH_VARARGS|METH_KEYWORDS,
    set_variance_thresholds.doc()
  },

  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMGMMMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMGMMMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMGMMMachine_Type.tp_name = GMMMachine_doc.name();
  PyBobLearnEMGMMMachine_Type.tp_basicsize = sizeof(PyBobLearnEMGMMMachineObject);
  PyBobLearnEMGMMMachine_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMGMMMachine_Type.tp_doc = GMMMachine_doc.doc();

  // set the functions
  PyBobLearnEMGMMMachine_Type.tp_new = PyType_GenericNew;
  PyBobLearnEMGMMMachine_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnEMGMMMachine_init);
  PyBobLearnEMGMMMachine_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnEMGMMMachine_delete);
  PyBobLearnEMGMMMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMGMMMachine_RichCompare);
  PyBobLearnEMGMMMachine_Type.tp_methods = PyBobLearnEMGMMMachine_methods;
  PyBobLearnEMGMMMachine_Type.tp_getset = PyBobLearnEMGMMMachine_getseters;
  PyBobLearnEMGMMMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMGMMMachine_loglikelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMGMMMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMGMMMachine_Type);
  return PyModule_AddObject(module, "GMMMachine", (PyObject*)&PyBobLearnEMGMMMachine_Type) >= 0;
}

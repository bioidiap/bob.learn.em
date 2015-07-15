/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Fri 26 Dec 16:18:00 2014
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto KMeansMachine_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".KMeansMachine",
  "This class implements a k-means classifier.\n"
  "See Section 9.1 of Bishop, \"Pattern recognition and machine learning\", 2006"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a KMeansMachine",
    "",
    true
  )
  .add_prototype("n_means,n_inputs","")
  .add_prototype("other","")
  .add_prototype("hdf5","")
  .add_prototype("","")

  .add_parameter("n_means", "int", "Number of means")
  .add_parameter("n_inputs", "int", "Dimension of the feature vector")
  .add_parameter("other", ":py:class:`bob.learn.em.KMeansMachine`", "A KMeansMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnEMKMeansMachine_init_number(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = KMeansMachine_doc.kwlist(0);
  int n_inputs    = 1;
  int n_means = 1;
  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &n_means, &n_inputs))
    return -1;

  if(n_means < 0){
    PyErr_Format(PyExc_TypeError, "means argument must be greater than or equal to zero");
    KMeansMachine_doc.print_usage();
    return -1;
  }

  if(n_inputs < 0){
    PyErr_Format(PyExc_TypeError, "input argument must be greater than or equal to zero");
    KMeansMachine_doc.print_usage();
    return -1;
   }

  self->cxx.reset(new bob::learn::em::KMeansMachine(n_means, n_inputs));
  return 0;
}


static int PyBobLearnEMKMeansMachine_init_copy(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = KMeansMachine_doc.kwlist(1);
  PyBobLearnEMKMeansMachineObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMKMeansMachine_Type, &tt)){
    KMeansMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::KMeansMachine(*tt->cxx));
  return 0;
}


static int PyBobLearnEMKMeansMachine_init_hdf5(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = KMeansMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    KMeansMachine_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);
  self->cxx.reset(new bob::learn::em::KMeansMachine(*(config->f)));

  return 0;
}


static int PyBobLearnEMKMeansMachine_init(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs) {

    case 0: //default initializer ()
      self->cxx.reset(new bob::learn::em::KMeansMachine());
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
     if (PyBobLearnEMKMeansMachine_Check(arg))
       return PyBobLearnEMKMeansMachine_init_copy(self, args, kwargs);
      // If the constructor input is a HDF5
     else if (PyBobIoHDF5File_Check(arg))
       return PyBobLearnEMKMeansMachine_init_hdf5(self, args, kwargs);
    }
    case 2:
      return PyBobLearnEMKMeansMachine_init_number(self, args, kwargs);
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0, 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      KMeansMachine_doc.print_usage();
      return -1;
  }
  BOB_CATCH_MEMBER("cannot create KMeansMachine", -1)
  return 0;
}



static void PyBobLearnEMKMeansMachine_delete(PyBobLearnEMKMeansMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMKMeansMachine_RichCompare(PyBobLearnEMKMeansMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMKMeansMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMKMeansMachineObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare KMeansMachine objects", 0)
}

int PyBobLearnEMKMeansMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMKMeansMachine_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int)",
  "A tuple that represents the number of means and dimensionality of the feature vector``(n_means, dim)``.",
  ""
);
PyObject* PyBobLearnEMKMeansMachine_getShape(PyBobLearnEMKMeansMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i)", self->cxx->getNMeans(), self->cxx->getNInputs());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}

/***** MEAN *****/

static auto means = bob::extension::VariableDoc(
  "means",
  "array_like <float, 2D>",
  "The means",
  ""
);
PyObject* PyBobLearnEMKMeansMachine_getMeans(PyBobLearnEMKMeansMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getMeans());
  BOB_CATCH_MEMBER("means could not be read", 0)
}
int PyBobLearnEMKMeansMachine_setMeans(PyBobLearnEMKMeansMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, means.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, means.name());
    return 0;
  }

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, means.name());
    return 0;
  }

  if (input->shape[1] != (Py_ssize_t)self->cxx->getNInputs()) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [N, %" PY_FORMAT_SIZE_T "d] not [N, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0], means.name());
    return 0;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "means");
  if (!b) return -1;
  self->cxx->setMeans(*b);
  return 0;
  BOB_CATCH_MEMBER("means could not be set", -1)
}


static PyGetSetDef PyBobLearnEMKMeansMachine_getseters[] = {
  {
   shape.name(),
   (getter)PyBobLearnEMKMeansMachine_getShape,
   0,
   shape.doc(),
   0
  },
  {
   means.name(),
   (getter)PyBobLearnEMKMeansMachine_getMeans,
   (setter)PyBobLearnEMKMeansMachine_setMeans,
   means.doc(),
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
  "Save the configuration of the KMeansMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnEMKMeansMachine_Save(PyBobLearnEMKMeansMachineObject* self,  PyObject* args, PyObject* kwargs) {

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
  "Load the configuration of the KMeansMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnEMKMeansMachine_Load(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
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

  "Compares this KMeansMachine with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.KMeansMachine`", "A KMeansMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMKMeansMachine_IsSimilarTo(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMKMeansMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMKMeansMachine_Type, &other,
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
.add_prototype("n_means,n_inputs")
.add_parameter("n_means", "int", "Number of means")
.add_parameter("n_inputs", "int", "Dimensionality of the feature vector");
static PyObject* PyBobLearnEMKMeansMachine_resize(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  int n_means = 0;
  int n_inputs = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &n_means, &n_inputs)) return 0;

  if (n_means <= 0){
    PyErr_Format(PyExc_TypeError, "n_means must be greater than zero");
    resize.print_usage();
    return 0;
  }
  if (n_inputs <= 0){
    PyErr_Format(PyExc_TypeError, "n_inputs must be greater than zero");
    resize.print_usage();
    return 0;
  }

  self->cxx->resize(n_means, n_inputs);

  BOB_CATCH_MEMBER("cannot perform the resize method", 0)

  Py_RETURN_NONE;
}

/*** get_mean ***/
static auto get_mean = bob::extension::FunctionDoc(
  "get_mean",
  "Get the i'th mean.",
  ".. note:: An exception is thrown if i is out of range.",
  true
)
.add_prototype("i","mean")
.add_parameter("i", "int", "Index of the mean")
.add_return("mean","array_like <float, 1D>","Mean array");
static PyObject* PyBobLearnEMKMeansMachine_get_mean(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_mean.kwlist(0);

  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) return 0;

  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getMean(i));

  BOB_CATCH_MEMBER("cannot get the mean", 0)
}


/*** set_mean ***/
static auto set_mean = bob::extension::FunctionDoc(
  "set_mean",
  "Set the i'th mean.",
  ".. note:: An exception is thrown if i is out of range.",
  true
)
.add_prototype("i,mean")
.add_parameter("i", "int", "Index of the mean")
.add_parameter("mean", "array_like <float, 1D>", "Mean array");
static PyObject* PyBobLearnEMKMeansMachine_set_mean(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = set_mean.kwlist(0);

  int i = 0;
  PyBlitzArrayObject* mean = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO&", kwlist, &i, &PyBlitzArray_Converter, &mean)) return 0;

  //protects acquired resources through this scope
  auto mean_ = make_safe(mean);

  // perform check on the input
  if (mean->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, set_mean.name());
    return 0;
  }

  if (mean->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, set_mean.name());
    return 0;
  }

  if (mean->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), mean->shape[0], set_mean.name());
    return 0;
  }

  //setting the mean
  self->cxx->setMean(i, *PyBlitzArrayCxx_AsBlitz<double,1>(mean));

  BOB_CATCH_MEMBER("cannot set the mean", 0)

  Py_RETURN_NONE;
}



/*** get_distance_from_mean ***/
static auto get_distance_from_mean = bob::extension::FunctionDoc(
  "get_distance_from_mean",
  "Return the power of two of the square Euclidean distance of the sample, x, to the i'th mean.",
  ".. note:: An exception is thrown if i is out of range.",
  true
)
.add_prototype("input,i","output")
.add_parameter("input", "array_like <float, 1D>", "The data sample (feature vector)")
.add_parameter("i", "int", "The index of the mean")
.add_return("output","float","Square Euclidean distance of the sample, x, to the i'th mean");
static PyObject* PyBobLearnEMKMeansMachine_get_distance_from_mean(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_distance_from_mean.kwlist(0);

  PyBlitzArrayObject* input = 0;
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&i", kwlist, &PyBlitzArray_Converter, &input, &i)){
    return 0;
  }

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, get_distance_from_mean.name());
    return 0;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, get_distance_from_mean.name());
    return 0;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0], get_distance_from_mean.name());
    return 0;
  }

  double output = self->cxx->getDistanceFromMean(*PyBlitzArrayCxx_AsBlitz<double,1>(input),i);
  return Py_BuildValue("d", output);

  BOB_CATCH_MEMBER("cannot compute the likelihood", 0)
}


/*** get_closest_mean ***/
static auto get_closest_mean = bob::extension::FunctionDoc(
  "get_closest_mean",
  "Calculate the index of the mean that is closest (in terms of square Euclidean distance) to the data sample, x.",
  "",
  true
)
.add_prototype("input","output")
.add_parameter("input", "array_like <float, 1D>", "The data sample (feature vector)")
.add_return("output", "(int, int)", "Tuple containing the closest mean and the minimum distance from the input");
static PyObject* PyBobLearnEMKMeansMachine_get_closest_mean(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_closest_mean.kwlist(0);

  PyBlitzArrayObject* input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  size_t closest_mean = 0;
  double min_distance = -1;

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, get_closest_mean.name());
    return 0;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, get_closest_mean.name());
    return 0;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0], get_closest_mean.name());
    return 0;
  }

  self->cxx->getClosestMean(*PyBlitzArrayCxx_AsBlitz<double,1>(input), closest_mean, min_distance);

  return Py_BuildValue("(i,d)", closest_mean, min_distance);

  BOB_CATCH_MEMBER("cannot compute the closest mean", 0)
}


/*** get_min_distance ***/
static auto get_min_distance = bob::extension::FunctionDoc(
  "get_min_distance",
  "Output the minimum (Square Euclidean) distance between the input and the closest mean ",
  "",
  true
)
.add_prototype("input","output")
.add_parameter("input", "array_like <float, 1D>", "The data sample (feature vector)")
.add_return("output", "float", "The minimum distance");
static PyObject* PyBobLearnEMKMeansMachine_get_min_distance(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = get_min_distance.kwlist(0);

  PyBlitzArrayObject* input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
  double min_distance = 0;

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, get_min_distance.name());
    return 0;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, get_min_distance.name());
    return 0;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNInputs()){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[0], get_min_distance.name());
    return 0;
  }

  min_distance = self->cxx->getMinDistance(*PyBlitzArrayCxx_AsBlitz<double,1>(input));

  return Py_BuildValue("d", min_distance);

  BOB_CATCH_MEMBER("cannot compute the min distance", 0)
}

/**** get_variances_and_weights_for_each_cluster ***/
static auto get_variances_and_weights_for_each_cluster = bob::extension::FunctionDoc(
  "get_variances_and_weights_for_each_cluster",
  "For each mean, find the subset of the samples that is closest to that mean, and calculate"
  " 1) the variance of that subset (the cluster variance)"
  " 2) the proportion of the samples represented by that subset (the cluster weight)",
  "",
  true
)
.add_prototype("input","output")
.add_parameter("input", "array_like <float, 2D>", "The data sample (feature vector)")
.add_return("output", "(array_like <float, 2D>, array_like <float, 1D>)", "A tuple with the variances and the weights respectively");
static PyObject* PyBobLearnEMKMeansMachine_get_variances_and_weights_for_each_cluster(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist =  get_variances_and_weights_for_each_cluster.kwlist(0);

  PyBlitzArrayObject* input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, get_variances_and_weights_for_each_cluster.name());
    return 0;
  }

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, get_variances_and_weights_for_each_cluster.name());
    return 0;
  }

  if (input->shape[1] != (Py_ssize_t)self->cxx->getNInputs() ) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [N, %" PY_FORMAT_SIZE_T "d] not [N, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), input->shape[1], get_variances_and_weights_for_each_cluster.name());
    return 0;
  }

  blitz::Array<double,2> variances(self->cxx->getNMeans(),self->cxx->getNInputs());
  blitz::Array<double,1> weights(self->cxx->getNMeans());

  self->cxx->getVariancesAndWeightsForEachCluster(*PyBlitzArrayCxx_AsBlitz<double,2>(input),variances,weights);

  return Py_BuildValue("(N,N)",PyBlitzArrayCxx_AsConstNumpy(variances), PyBlitzArrayCxx_AsConstNumpy(weights));

  BOB_CATCH_MEMBER("cannot compute the variances and weights for each cluster", 0)
}


/**** __get_variances_and_weights_for_each_cluster_init__ ***/
static auto __get_variances_and_weights_for_each_cluster_init__ = bob::extension::FunctionDoc(
  "__get_variances_and_weights_for_each_cluster_init__",
  "Methods consecutively called by getVariancesAndWeightsForEachCluster()"
  "This should help for the parallelization on several nodes by splitting the data and calling"
  "getVariancesAndWeightsForEachClusterAcc() for each split. In this case, there is a need to sum"
  "with the m_cache_means, variances, and weights variables before performing the merge on one"
  "node using getVariancesAndWeightsForEachClusterFin().",
  "",
  true
)
.add_prototype("variances,weights","")
.add_parameter("variances", "array_like <float, 2D>", "Variance array")
.add_parameter("weights", "array_like <float, 1D>", "Weight array");
static PyObject* PyBobLearnEMKMeansMachine_get_variances_and_weights_for_each_cluster_init(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist =  __get_variances_and_weights_for_each_cluster_init__.kwlist(0);

  PyBlitzArrayObject* variances = 0;
  PyBlitzArrayObject* weights   = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&", kwlist, &PyBlitzArray_Converter, &variances,  &PyBlitzArray_Converter, &weights)) return 0;

  //protects acquired resources through this scope
  auto weights_   = make_safe(weights);
  auto variances_ = make_safe(variances);

  self->cxx->getVariancesAndWeightsForEachClusterInit(*PyBlitzArrayCxx_AsBlitz<double,2>(variances), *PyBlitzArrayCxx_AsBlitz<double,1>(weights));
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot compute the variances and weights for each cluster", 0)
}


/**** __get_variances_and_weights_for_each_cluster_acc__ ***/
static auto __get_variances_and_weights_for_each_cluster_acc__ = bob::extension::FunctionDoc(
  "__get_variances_and_weights_for_each_cluster_acc__",
  "Methods consecutively called by getVariancesAndWeightsForEachCluster()"
  "This should help for the parallelization on several nodes by splitting the data and calling"
  "getVariancesAndWeightsForEachClusterAcc() for each split. In this case, there is a need to sum"
  "with the m_cache_means, variances, and weights variables before performing the merge on one"
  "node using getVariancesAndWeightsForEachClusterFin().",
  "",
  true
)
.add_prototype("data,variances,weights","")
.add_parameter("data", "array_like <float, 2D>", "data array")
.add_parameter("variances", "array_like <float, 2D>", "Variance array")
.add_parameter("weights", "array_like <float, 1D>", "Weight array");
static PyObject* PyBobLearnEMKMeansMachine_get_variances_and_weights_for_each_cluster_acc(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist =  __get_variances_and_weights_for_each_cluster_acc__.kwlist(0);

  PyBlitzArrayObject* data      = 0;
  PyBlitzArrayObject* variances = 0;
  PyBlitzArrayObject* weights   = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&", kwlist, &PyBlitzArray_Converter, &data, &PyBlitzArray_Converter, &variances, &PyBlitzArray_Converter, &weights)) return 0;

  //protects acquired resources through this scope
  auto data_      = make_safe(data);
  auto weights_   = make_safe(weights);
  auto variances_ = make_safe(variances);

  self->cxx->getVariancesAndWeightsForEachClusterAcc(*PyBlitzArrayCxx_AsBlitz<double,2>(data), *PyBlitzArrayCxx_AsBlitz<double,2>(variances), *PyBlitzArrayCxx_AsBlitz<double,1>(weights));
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot compute the variances and weights for each cluster", 0)
}


/**** __get_variances_and_weights_for_each_cluster_fin__ ***/
static auto __get_variances_and_weights_for_each_cluster_fin__ = bob::extension::FunctionDoc(
  "__get_variances_and_weights_for_each_cluster_fin__",
  "Methods consecutively called by getVariancesAndWeightsForEachCluster()"
  "This should help for the parallelization on several nodes by splitting the data and calling"
  "getVariancesAndWeightsForEachClusterAcc() for each split. In this case, there is a need to sum"
  "with the m_cache_means, variances, and weights variables before performing the merge on one"
  "node using getVariancesAndWeightsForEachClusterFin().",
  "",
  true
)
.add_prototype("variances,weights","")
.add_parameter("variances", "array_like <float, 2D>", "Variance array")
.add_parameter("weights", "array_like <float, 1D>", "Weight array");
static PyObject* PyBobLearnEMKMeansMachine_get_variances_and_weights_for_each_cluster_fin(PyBobLearnEMKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist =  __get_variances_and_weights_for_each_cluster_fin__.kwlist(0);

  PyBlitzArrayObject* variances = 0;
  PyBlitzArrayObject* weights   = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&", kwlist, &PyBlitzArray_Converter, &variances,  &PyBlitzArray_Converter, &weights)) return 0;

  //protects acquired resources through this scope
  auto weights_   = make_safe(weights);
  auto variances_ = make_safe(variances);

  self->cxx->getVariancesAndWeightsForEachClusterFin(*PyBlitzArrayCxx_AsBlitz<double,2>(variances), *PyBlitzArrayCxx_AsBlitz<double,1>(weights));
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("cannot compute the variances and weights for each cluster", 0)
}


static PyMethodDef PyBobLearnEMKMeansMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  {
    get_mean.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_get_mean,
    METH_VARARGS|METH_KEYWORDS,
    get_mean.doc()
  },
  {
    set_mean.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_set_mean,
    METH_VARARGS|METH_KEYWORDS,
    set_mean.doc()
  },
  {
    get_distance_from_mean.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_get_distance_from_mean,
    METH_VARARGS|METH_KEYWORDS,
    get_distance_from_mean.doc()
  },
  {
    get_closest_mean.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_get_closest_mean,
    METH_VARARGS|METH_KEYWORDS,
    get_closest_mean.doc()
  },
  {
    get_min_distance.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_get_min_distance,
    METH_VARARGS|METH_KEYWORDS,
    get_min_distance.doc()
  },
  {
    get_variances_and_weights_for_each_cluster.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_get_variances_and_weights_for_each_cluster,
    METH_VARARGS|METH_KEYWORDS,
    get_variances_and_weights_for_each_cluster.doc()
  },
  {
    __get_variances_and_weights_for_each_cluster_init__.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_get_variances_and_weights_for_each_cluster_init,
    METH_VARARGS|METH_KEYWORDS,
    __get_variances_and_weights_for_each_cluster_init__.doc()
  },
  {
    __get_variances_and_weights_for_each_cluster_acc__.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_get_variances_and_weights_for_each_cluster_acc,
    METH_VARARGS|METH_KEYWORDS,
    __get_variances_and_weights_for_each_cluster_acc__.doc()
  },
  {
    __get_variances_and_weights_for_each_cluster_fin__.name(),
    (PyCFunction)PyBobLearnEMKMeansMachine_get_variances_and_weights_for_each_cluster_fin,
    METH_VARARGS|METH_KEYWORDS,
    __get_variances_and_weights_for_each_cluster_fin__.doc()
  },

  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMKMeansMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMKMeansMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMKMeansMachine_Type.tp_name = KMeansMachine_doc.name();
  PyBobLearnEMKMeansMachine_Type.tp_basicsize = sizeof(PyBobLearnEMKMeansMachineObject);
  PyBobLearnEMKMeansMachine_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMKMeansMachine_Type.tp_doc = KMeansMachine_doc.doc();

  // set the functions
  PyBobLearnEMKMeansMachine_Type.tp_new = PyType_GenericNew;
  PyBobLearnEMKMeansMachine_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnEMKMeansMachine_init);
  PyBobLearnEMKMeansMachine_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnEMKMeansMachine_delete);
  PyBobLearnEMKMeansMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMKMeansMachine_RichCompare);
  PyBobLearnEMKMeansMachine_Type.tp_methods = PyBobLearnEMKMeansMachine_methods;
  PyBobLearnEMKMeansMachine_Type.tp_getset = PyBobLearnEMKMeansMachine_getseters;
  //PyBobLearnEMGMMMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMGMMMachine_loglikelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMKMeansMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMKMeansMachine_Type);
  return PyModule_AddObject(module, "KMeansMachine", (PyObject*)&PyBobLearnEMKMeansMachine_Type) >= 0;
}

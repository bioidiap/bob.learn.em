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
  .add_prototype("n_gaussians,n_inputs","")
  .add_prototype("other","")
  .add_prototype("hdf5","")
  .add_prototype("","")

  .add_parameter("n_means", "int", "Number of means")
  .add_parameter("n_inputs", "int", "Dimension of the feature vector")
  .add_parameter("other", ":py:class:`bob.learn.misc.KMeansMachine`", "A KMeansMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnMiscKMeansMachine_init_number(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {

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

  self->cxx.reset(new bob::learn::misc::KMeansMachine(n_means, n_inputs));
  return 0;
}


static int PyBobLearnMiscKMeansMachine_init_copy(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = KMeansMachine_doc.kwlist(1);
  PyBobLearnMiscKMeansMachineObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscKMeansMachine_Type, &tt)){
    KMeansMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::KMeansMachine(*tt->cxx));
  return 0;
}


static int PyBobLearnMiscKMeansMachine_init_hdf5(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = KMeansMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    KMeansMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::KMeansMachine(*(config->f)));

  return 0;
}


static int PyBobLearnMiscKMeansMachine_init(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);
  
  switch (nargs) {

    case 0: //default initializer ()
      self->cxx.reset(new bob::learn::misc::KMeansMachine());
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
     if (PyBobLearnMiscKMeansMachine_Check(arg))
       return PyBobLearnMiscKMeansMachine_init_copy(self, args, kwargs);
      // If the constructor input is a HDF5
     else if (PyBobIoHDF5File_Check(arg))
       return PyBobLearnMiscKMeansMachine_init_hdf5(self, args, kwargs);
    }
    case 2:
      return PyBobLearnMiscKMeansMachine_init_number(self, args, kwargs);
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0, 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      KMeansMachine_doc.print_usage();
      return -1;
  }
  BOB_CATCH_MEMBER("cannot create KMeansMachine", 0)
  return 0;
}



static void PyBobLearnMiscKMeansMachine_delete(PyBobLearnMiscKMeansMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnMiscKMeansMachine_RichCompare(PyBobLearnMiscKMeansMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscKMeansMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscKMeansMachineObject*>(other);
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

int PyBobLearnMiscKMeansMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscKMeansMachine_Type));
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
PyObject* PyBobLearnMiscKMeansMachine_getShape(PyBobLearnMiscKMeansMachineObject* self, void*) {
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
PyObject* PyBobLearnMiscKMeansMachine_getMeans(PyBobLearnMiscKMeansMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getMeans());
  BOB_CATCH_MEMBER("means could not be read", 0)
}
int PyBobLearnMiscKMeansMachine_setMeans(PyBobLearnMiscKMeansMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, means.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(o, "means");
  if (!b) return -1;
  self->cxx->setMeans(*b);
  return 0;
  BOB_CATCH_MEMBER("means could not be set", -1)
}


static PyGetSetDef PyBobLearnMiscKMeansMachine_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnMiscKMeansMachine_getShape,
   0,
   shape.doc(),
   0
  },
  {
   means.name(),
   (getter)PyBobLearnMiscKMeansMachine_getMeans,
   (setter)PyBobLearnMiscKMeansMachine_setMeans,
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
static PyObject* PyBobLearnMiscKMeansMachine_Save(PyBobLearnMiscKMeansMachineObject* self,  PyObject* args, PyObject* kwargs) {

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
static PyObject* PyBobLearnMiscKMeansMachine_Load(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
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
.add_parameter("other", ":py:class:`bob.learn.misc.KMeansMachine`", "A KMeansMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnMiscKMeansMachine_IsSimilarTo(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnMiscKMeansMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnMiscKMeansMachine_Type, &other,
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
static PyObject* PyBobLearnMiscKMeansMachine_resize(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  int n_means = 0;
  int n_inputs = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &n_means, &n_inputs)) Py_RETURN_NONE;

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
.add_prototype("i")
.add_parameter("i", "int", "Index of the mean")
.add_return("mean","array_like <float, 1D>","Mean array");
static PyObject* PyBobLearnMiscKMeansMachine_get_mean(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_mean.kwlist(0);

  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;
 
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
static PyObject* PyBobLearnMiscKMeansMachine_set_mean(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = set_mean.kwlist(0);

  int i = 0;
  PyBlitzArrayObject* mean = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO&", kwlist, &i, &PyBlitzArray_Converter, &mean)) Py_RETURN_NONE;
  
  //protects acquired resources through this scope
  auto mean_ = make_safe(mean);

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
static PyObject* PyBobLearnMiscKMeansMachine_get_distance_from_mean(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_distance_from_mean.kwlist(0);

  PyBlitzArrayObject* input = 0;
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&i", kwlist, &PyBlitzArray_Converter, &input, &i)){ 
    Py_RETURN_NONE;
  }

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

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
static PyObject* PyBobLearnMiscKMeansMachine_get_closest_mean(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_closest_mean.kwlist(0);

  PyBlitzArrayObject* input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) Py_RETURN_NONE;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  size_t closest_mean = 0;
  double min_distance = -1;   
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
.add_return("output", "double", "The minimum distance");
static PyObject* PyBobLearnMiscKMeansMachine_get_min_distance(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_min_distance.kwlist(0);

  PyBlitzArrayObject* input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) Py_RETURN_NONE;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  double min_distance = 0;   
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
static PyObject* PyBobLearnMiscKMeansMachine_get_variances_and_weights_for_each_cluster(PyBobLearnMiscKMeansMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist =  get_variances_and_weights_for_each_cluster.kwlist(0);

  PyBlitzArrayObject* input = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) Py_RETURN_NONE;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  blitz::Array<double,2> variances(self->cxx->getNMeans(),self->cxx->getNInputs());
  blitz::Array<double,1> weights(self->cxx->getNMeans());
  
  self->cxx->getVariancesAndWeightsForEachCluster(*PyBlitzArrayCxx_AsBlitz<double,2>(input),variances,weights);

  return Py_BuildValue("(O,O)",PyBlitzArrayCxx_AsConstNumpy(variances), PyBlitzArrayCxx_AsConstNumpy(weights));

  BOB_CATCH_MEMBER("cannot compute the variances and weights for each cluster", 0)
}



static PyMethodDef PyBobLearnMiscKMeansMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },  
  {
    get_mean.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_get_mean,
    METH_VARARGS|METH_KEYWORDS,
    get_mean.doc()
  },  
  {
    set_mean.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_set_mean,
    METH_VARARGS|METH_KEYWORDS,
    set_mean.doc()
  },  
  {
    get_distance_from_mean.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_get_distance_from_mean,
    METH_VARARGS|METH_KEYWORDS,
    get_distance_from_mean.doc()
  },  
  {
    get_closest_mean.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_get_closest_mean,
    METH_VARARGS|METH_KEYWORDS,
    get_closest_mean.doc()
  },  
  {
    get_min_distance.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_get_min_distance,
    METH_VARARGS|METH_KEYWORDS,
    get_min_distance.doc()
  },  

  {
    get_variances_and_weights_for_each_cluster.name(),
    (PyCFunction)PyBobLearnMiscKMeansMachine_get_variances_and_weights_for_each_cluster,
    METH_VARARGS|METH_KEYWORDS,
    get_variances_and_weights_for_each_cluster.doc()
  },  

  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscKMeansMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscKMeansMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscKMeansMachine_Type.tp_name = KMeansMachine_doc.name();
  PyBobLearnMiscKMeansMachine_Type.tp_basicsize = sizeof(PyBobLearnMiscKMeansMachineObject);
  PyBobLearnMiscKMeansMachine_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscKMeansMachine_Type.tp_doc = KMeansMachine_doc.doc();

  // set the functions
  PyBobLearnMiscKMeansMachine_Type.tp_new = PyType_GenericNew;
  PyBobLearnMiscKMeansMachine_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnMiscKMeansMachine_init);
  PyBobLearnMiscKMeansMachine_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnMiscKMeansMachine_delete);
  PyBobLearnMiscKMeansMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscKMeansMachine_RichCompare);
  PyBobLearnMiscKMeansMachine_Type.tp_methods = PyBobLearnMiscKMeansMachine_methods;
  PyBobLearnMiscKMeansMachine_Type.tp_getset = PyBobLearnMiscKMeansMachine_getseters;
  //PyBobLearnMiscGMMMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnMiscGMMMachine_loglikelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscKMeansMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscKMeansMachine_Type);
  return PyModule_AddObject(module, "KMeansMachine", (PyObject*)&PyBobLearnMiscKMeansMachine_Type) >= 0;
}


/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Wed 03 Dec 14:38:48 2014
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto GMMStats_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".GMMStats",
  "A container for GMM statistics",
  "With respect to [Reynolds2000]_ the class computes: \n\n"
  "* Eq (8) is :py:class:`bob.learn.em.GMMStats.n`: :math:`n_i=\\sum\\limits_{t=1}^T Pr(i | x_t)`\n\n"
  "* Eq (9) is :py:class:`bob.learn.em.GMMStats.sum_px`:  :math:`E_i(x)=\\frac{1}{n(i)}\\sum\\limits_{t=1}^T Pr(i | x_t)x_t`\n\n"
  "* Eq (10) is :py:class:`bob.learn.em.GMMStats.sum_pxx`: :math:`E_i(x^2)=\\frac{1}{n(i)}\\sum\\limits_{t=1}^T Pr(i | x_t)x_t^2`\n\n"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "A container for GMM statistics.",
    "",
    true
  )
  .add_prototype("n_gaussians,n_inputs","")
  .add_prototype("other","")
  .add_prototype("hdf5","")
  .add_prototype("","")

  .add_parameter("n_gaussians", "int", "Number of gaussians")
  .add_parameter("n_inputs", "int", "Dimension of the feature vector")
  .add_parameter("other", ":py:class:`bob.learn.em.GMMStats`", "A GMMStats object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnEMGMMStats_init_number(PyBobLearnEMGMMStatsObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMStats_doc.kwlist(0);
  int n_inputs    = 1;
  int n_gaussians = 1;
  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &n_gaussians, &n_inputs))
    return -1;

  if(n_gaussians < 0){
    PyErr_Format(PyExc_TypeError, "gaussians argument must be greater than or equal to zero");
    GMMStats_doc.print_usage();
    return -1;
  }

  if(n_inputs < 0){
    PyErr_Format(PyExc_TypeError, "input argument must be greater than or equal to zero");
    GMMStats_doc.print_usage();
    return -1;
   }

  self->cxx.reset(new bob::learn::em::GMMStats(n_gaussians, n_inputs));
  return 0;
}


static int PyBobLearnEMGMMStats_init_copy(PyBobLearnEMGMMStatsObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMStats_doc.kwlist(1);
  PyBobLearnEMGMMStatsObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMGMMStats_Type, &tt)){
    GMMStats_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::GMMStats(*tt->cxx));
  return 0;
}


static int PyBobLearnEMGMMStats_init_hdf5(PyBobLearnEMGMMStatsObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMStats_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    GMMStats_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);
  self->cxx.reset(new bob::learn::em::GMMStats(*(config->f)));

  return 0;
}



static int PyBobLearnEMGMMStats_init(PyBobLearnEMGMMStatsObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs) {

    case 0: //default initializer ()
      self->cxx.reset(new bob::learn::em::GMMStats());
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

      /**If the constructor input is Gaussian object**/
     if (PyBobLearnEMGMMStats_Check(arg))
       return PyBobLearnEMGMMStats_init_copy(self, args, kwargs);
      /**If the constructor input is a HDF5**/
     else if (PyBobIoHDF5File_Check(arg))
       return PyBobLearnEMGMMStats_init_hdf5(self, args, kwargs);
    }
    case 2:
      return PyBobLearnEMGMMStats_init_number(self, args, kwargs);
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0, 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      GMMStats_doc.print_usage();
      return -1;
  }
  BOB_CATCH_MEMBER("cannot create GMMStats", -1)
  return 0;
}



static void PyBobLearnEMGMMStats_delete(PyBobLearnEMGMMStatsObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMGMMStats_RichCompare(PyBobLearnEMGMMStatsObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMGMMStats_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMGMMStatsObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare GMMStats objects", 0)
}

int PyBobLearnEMGMMStats_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMGMMStats_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** n *****/
static auto n = bob::extension::VariableDoc(
  "n",
  "array_like <float, 1D>",
  "For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of :math:`P(gaussian_i|x)`"
);
PyObject* PyBobLearnEMGMMStats_getN(PyBobLearnEMGMMStatsObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->n);
  BOB_CATCH_MEMBER("n could not be read", 0)
}
int PyBobLearnEMGMMStats_setN(PyBobLearnEMGMMStatsObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, n.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, n.name());
    return -1;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, n.name());
    return -1;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->n.extent(0)){
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->n.extent(0), input->shape[0], n.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "n");
  if (!b) return -1;
  self->cxx->n = *b;
  return 0;
  BOB_CATCH_MEMBER("n could not be set", -1)
}


/***** sum_px *****/
static auto sum_px = bob::extension::VariableDoc(
  "sum_px",
  "array_like <float, 2D>",
  "For each Gaussian, the accumulated sum of responsibility times the sample"
);
PyObject* PyBobLearnEMGMMStats_getSum_px(PyBobLearnEMGMMStatsObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->sumPx);
  BOB_CATCH_MEMBER("sum_px could not be read", 0)
}
int PyBobLearnEMGMMStats_setSum_px(PyBobLearnEMGMMStatsObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, sum_px.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, sum_px.name());
    return -1;
  }

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, sum_px.name());
    return -1;
  }

  if (input->shape[1] != (Py_ssize_t)self->cxx->sumPx.extent(1) && input->shape[0] != (Py_ssize_t)self->cxx->sumPx.extent(0)) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->sumPx.extent(1), (Py_ssize_t)self->cxx->sumPx.extent(0), (Py_ssize_t)input->shape[1], (Py_ssize_t)input->shape[0], sum_px.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "sum_px");
  if (!b) return -1;
  self->cxx->sumPx = *b;
  return 0;
  BOB_CATCH_MEMBER("sum_px could not be set", -1)
}


/***** sum_pxx *****/
static auto sum_pxx = bob::extension::VariableDoc(
  "sum_pxx",
  "array_like <float, 2D>",
  "For each Gaussian, the accumulated sum of responsibility times the sample squared"
);
PyObject* PyBobLearnEMGMMStats_getSum_pxx(PyBobLearnEMGMMStatsObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->sumPxx);
  BOB_CATCH_MEMBER("sum_pxx could not be read", 0)
}
int PyBobLearnEMGMMStats_setSum_pxx(PyBobLearnEMGMMStatsObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, sum_pxx.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, sum_pxx.name());
    return -1;
  }

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, sum_pxx.name());
    return -1;
  }

  if (input->shape[1] != (Py_ssize_t)self->cxx->sumPxx.extent(1) && input->shape[0] != (Py_ssize_t)self->cxx->sumPxx.extent(0)) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->sumPxx.extent(1), (Py_ssize_t)self->cxx->sumPxx.extent(0), (Py_ssize_t)input->shape[1], (Py_ssize_t)input->shape[0], sum_pxx.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "sum_pxx");
  if (!b) return -1;
  self->cxx->sumPxx = *b;
  return 0;
  BOB_CATCH_MEMBER("sum_pxx could not be set", -1)
}


/***** t *****/
static auto t = bob::extension::VariableDoc(
  "t",
  "int",
  "The number of samples"
);
PyObject* PyBobLearnEMGMMStats_getT(PyBobLearnEMGMMStatsObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->T);
  BOB_CATCH_MEMBER("t could not be read", 0)
}
int PyBobLearnEMGMMStats_setT(PyBobLearnEMGMMStatsObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyInt_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an int", Py_TYPE(self)->tp_name, t.name());
    return -1;
  }

  if (PyInt_AS_LONG(value) < 0){
    PyErr_Format(PyExc_TypeError, "t must be greater than or equal to zero");
    return -1;
  }

  self->cxx->T = PyInt_AS_LONG(value);
  BOB_CATCH_MEMBER("t could not be set", -1)
  return 0;
}


/***** log_likelihood *****/
static auto log_likelihood = bob::extension::VariableDoc(
  "log_likelihood",
  "float",
  "The accumulated log likelihood of all samples"
);
PyObject* PyBobLearnEMGMMStats_getLog_likelihood(PyBobLearnEMGMMStatsObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->log_likelihood);
  BOB_CATCH_MEMBER("log_likelihood could not be read", 0)
}
int PyBobLearnEMGMMStats_setLog_likelihood(PyBobLearnEMGMMStatsObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBob_NumberCheck(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an double", Py_TYPE(self)->tp_name, t.name());
    return -1;
  }

  self->cxx->log_likelihood = PyFloat_AsDouble(value);
  return 0;
  BOB_CATCH_MEMBER("log_likelihood could not be set", -1)
}


/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int)",
  "A tuple that represents the number of gaussians and dimensionality of each Gaussian ``(n_gaussians, dim)``.",
  ""
);
PyObject* PyBobLearnEMGMMStats_getShape(PyBobLearnEMGMMStatsObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i)", self->cxx->sumPx.shape()[0], self->cxx->sumPx.shape()[1]);
  BOB_CATCH_MEMBER("shape could not be read", 0)
}



static PyGetSetDef PyBobLearnEMGMMStats_getseters[] = {
  {
    n.name(),
    (getter)PyBobLearnEMGMMStats_getN,
    (setter)PyBobLearnEMGMMStats_setN,
    n.doc(),
    0
  },
  {
    sum_px.name(),
    (getter)PyBobLearnEMGMMStats_getSum_px,
    (setter)PyBobLearnEMGMMStats_setSum_px,
    sum_px.doc(),
    0
  },
  {
    sum_pxx.name(),
    (getter)PyBobLearnEMGMMStats_getSum_pxx,
    (setter)PyBobLearnEMGMMStats_setSum_pxx,
    sum_pxx.doc(),
    0
  },
  {
    t.name(),
    (getter)PyBobLearnEMGMMStats_getT,
    (setter)PyBobLearnEMGMMStats_setT,
    t.doc(),
    0
  },
  {
    log_likelihood.name(),
    (getter)PyBobLearnEMGMMStats_getLog_likelihood,
    (setter)PyBobLearnEMGMMStats_setLog_likelihood,
    log_likelihood.doc(),
    0
  },
  {
   shape.name(),
   (getter)PyBobLearnEMGMMStats_getShape,
   0,
   shape.doc(),
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
  "Save the configuration of the GMMStats to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnEMGMMStats_Save(PyBobLearnEMGMMStatsObject* self,  PyObject* args, PyObject* kwargs) {

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
  "Load the configuration of the GMMStats to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnEMGMMStats_Load(PyBobLearnEMGMMStatsObject* self, PyObject* args, PyObject* kwargs) {
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

  "Compares this GMMStats with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.GMMStats`", "A GMMStats object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMGMMStats_IsSimilarTo(PyBobLearnEMGMMStatsObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMGMMStatsObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMGMMStats_Type, &other,
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
static PyObject* PyBobLearnEMGMMStats_resize(PyBobLearnEMGMMStatsObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  int n_gaussians = 0;
  int n_inputs = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &n_gaussians, &n_inputs)) return 0;

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


/*** init ***/
static auto init = bob::extension::FunctionDoc(
  "init",
  " Resets statistics to zero."
)
.add_prototype("");
static PyObject* PyBobLearnEMGMMStats_init_method(PyBobLearnEMGMMStatsObject* self) {
  BOB_TRY

  self->cxx->init();

  BOB_CATCH_MEMBER("cannot perform the init method", 0)

  Py_RETURN_NONE;
}



static PyMethodDef PyBobLearnEMGMMStats_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnEMGMMStats_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMGMMStats_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMGMMStats_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnEMGMMStats_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  {
    init.name(),
    (PyCFunction)PyBobLearnEMGMMStats_init_method,
    METH_NOARGS,
    init.doc()
  },

  {0} /* Sentinel */
};


/******************************************************************/
/************ Operators *******************************************/
/******************************************************************/

static PyBobLearnEMGMMStatsObject* PyBobLearnEMGMMStats_inplaceadd(PyBobLearnEMGMMStatsObject* self, PyObject* other) {
  BOB_TRY

  if (!PyBobLearnEMGMMStats_Check(other)){
    PyErr_Format(PyExc_TypeError, "expected bob.learn.em.GMMStats object");
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnEMGMMStatsObject*>(other);

  self->cxx->operator+=(*other_->cxx);

  BOB_CATCH_MEMBER("it was not possible to process the operator +=", 0)

  Py_INCREF(self);
  return self;
}

static PyNumberMethods PyBobLearnEMGMMStats_operators = {0};

/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnEMGMMStats_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMGMMStats(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMGMMStats_Type.tp_name = GMMStats_doc.name();
  PyBobLearnEMGMMStats_Type.tp_basicsize = sizeof(PyBobLearnEMGMMStatsObject);
  PyBobLearnEMGMMStats_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMGMMStats_Type.tp_doc = GMMStats_doc.doc();

  // set the functions
  PyBobLearnEMGMMStats_Type.tp_new = PyType_GenericNew;
  PyBobLearnEMGMMStats_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnEMGMMStats_init);
  PyBobLearnEMGMMStats_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnEMGMMStats_delete);
  PyBobLearnEMGMMStats_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMGMMStats_RichCompare);
  PyBobLearnEMGMMStats_Type.tp_methods = PyBobLearnEMGMMStats_methods;
  PyBobLearnEMGMMStats_Type.tp_getset = PyBobLearnEMGMMStats_getseters;
  PyBobLearnEMGMMStats_Type.tp_call = 0;
  PyBobLearnEMGMMStats_Type.tp_as_number = &PyBobLearnEMGMMStats_operators;

  //set operators
  PyBobLearnEMGMMStats_operators.nb_inplace_add = reinterpret_cast<binaryfunc>(PyBobLearnEMGMMStats_inplaceadd);

  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMGMMStats_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMGMMStats_Type);
  return PyModule_AddObject(module, "GMMStats", (PyObject*)&PyBobLearnEMGMMStats_Type) >= 0;
}

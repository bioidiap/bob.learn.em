/**
 * @date Wed Jan 28 17:03:15 2015 +0200
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

static auto JFAMachine_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".JFAMachine",
  "A JFAMachine. An attached :py:class:`bob.learn.em.JFABase` should be provided for Joint Factor Analysis. The :py:class:`bob.learn.em.JFAMachine` carries information about the speaker factors :math:`y` and :math:`z`, whereas a :py:class:`bob.learn.em.JFABase` carries information about the matrices :math:`U`, :math:`V` and :math:`D`.\n\n"
  "References: [Vogt2008]_ [McCool2013]_",
  ""
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructor. Builds a new JFAMachine",
    "",
    true
  )
  .add_prototype("jfa_base","")
  .add_prototype("other","")
  .add_prototype("hdf5","")

  .add_parameter("jfa_base", ":py:class:`bob.learn.em.JFABase`", "The JFABase associated with this machine")
  .add_parameter("other", ":py:class:`bob.learn.em.JFAMachine`", "A JFAMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnEMJFAMachine_init_copy(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFAMachine_doc.kwlist(1);
  PyBobLearnEMJFAMachineObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMJFAMachine_Type, &o)){
    JFAMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::JFAMachine(*o->cxx));
  return 0;
}


static int PyBobLearnEMJFAMachine_init_hdf5(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFAMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    JFAMachine_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);
  self->cxx.reset(new bob::learn::em::JFAMachine(*(config->f)));

  return 0;
}


static int PyBobLearnEMJFAMachine_init_jfabase(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFAMachine_doc.kwlist(0);

  PyBobLearnEMJFABaseObject* jfa_base;

  //Here we have to select which keyword argument to read
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMJFABase_Type, &jfa_base)){
    JFAMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::JFAMachine(jfa_base->cxx));
  return 0;
}


static int PyBobLearnEMJFAMachine_init(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if(nargs == 1){
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
    if (PyBobLearnEMJFAMachine_Check(arg))
      return PyBobLearnEMJFAMachine_init_copy(self, args, kwargs);
    // If the constructor input is a HDF5
    else if (PyBobIoHDF5File_Check(arg))
      return PyBobLearnEMJFAMachine_init_hdf5(self, args, kwargs);
    // If the constructor input is a JFABase Object
    else
      return PyBobLearnEMJFAMachine_init_jfabase(self, args, kwargs);
  }
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires only 1 argument, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
    JFAMachine_doc.print_usage();
    return -1;
  }

  BOB_CATCH_MEMBER("cannot create JFAMachine", -1)
  return 0;
}

static void PyBobLearnEMJFAMachine_delete(PyBobLearnEMJFAMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMJFAMachine_RichCompare(PyBobLearnEMJFAMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMJFAMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMJFAMachineObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare JFAMachine objects", 0)
}

int PyBobLearnEMJFAMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMJFAMachine_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int, int, int)",
  "A tuple that represents the number of gaussians, dimensionality of each Gaussian, dimensionality of the rU (within client variability matrix) and dimensionality of the rV (between client variability matrix) ``(#Gaussians, #Inputs, #rU, #rV)``.",
  ""
);
PyObject* PyBobLearnEMJFAMachine_getShape(PyBobLearnEMJFAMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i,i,i)", self->cxx->getNGaussians(), self->cxx->getNInputs(), self->cxx->getDimRu(), self->cxx->getDimRv());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}

/***** supervector_length *****/
static auto supervector_length = bob::extension::VariableDoc(
  "supervector_length",
  "int",

  "Returns the supervector length.",
  "NGaussians x NInputs: Number of Gaussian components by the feature dimensionality"
  "@warning An exception is thrown if no Universal Background Model has been set yet."
  ""
);
PyObject* PyBobLearnEMJFAMachine_getSupervectorLength(PyBobLearnEMJFAMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getSupervectorLength());
  BOB_CATCH_MEMBER("supervector_length could not be read", 0)
}


/***** y *****/
static auto Y = bob::extension::VariableDoc(
  "y",
  "array_like <float, 1D>",
  "Returns the :math:`y` speaker factor. Eq (30) from [McCool2013]_",
  ""
);
PyObject* PyBobLearnEMJFAMachine_getY(PyBobLearnEMJFAMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getY());
  BOB_CATCH_MEMBER("`y` could not be read", 0)
}
int PyBobLearnEMJFAMachine_setY(PyBobLearnEMJFAMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, Y.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, Y.name());
    return -1;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, Y.name());
    return -1;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getY().extent(0)) {
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d, elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getY().extent(0), (Py_ssize_t)input->shape[0], Y.name());
    return -1;
  }


  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "y");
  if (!b) return -1;
  self->cxx->setY(*b);
  return 0;
  BOB_CATCH_MEMBER("`y` vector could not be set", -1)
}


/***** z *****/
static auto Z = bob::extension::VariableDoc(
  "z",
  "array_like <float, 1D>",
  "Returns the :math:`z` speaker factor. Eq (31) from [McCool2013]_",
  ""
);
PyObject* PyBobLearnEMJFAMachine_getZ(PyBobLearnEMJFAMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZ());
  BOB_CATCH_MEMBER("`z` could not be read", 0)
}
int PyBobLearnEMJFAMachine_setZ(PyBobLearnEMJFAMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, Z.name());
    return -1;
  }
  auto o_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, Z.name());
    return -1;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, Z.name());
    return -1;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getZ().extent(0)) {
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d, elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getZ().extent(0), (Py_ssize_t)input->shape[0], Z.name());
    return -1;
  }

  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "z");
  if (!b) return -1;
  self->cxx->setZ(*b);
  return 0;
  BOB_CATCH_MEMBER("`z` vector could not be set", -1)
}


/***** x *****/
static auto X = bob::extension::VariableDoc(
  "x",
  "array_like <float, 1D>",
  "Returns the :math:`X` session factor. Eq (29) from [McCool2013]_",
  "The latent variable :math:`x` (last one computed). This is a feature provided for convenience, but this attribute is not 'part' of the machine. The session latent variable :math:`x` is indeed not class-specific, but depends on the sample considered. Furthermore, it is not saved into the machine or used when comparing machines."
);
PyObject* PyBobLearnEMJFAMachine_getX(PyBobLearnEMJFAMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getX());
  BOB_CATCH_MEMBER("`x` could not be read", 0)
}


/***** jfa_base *****/
static auto jfa_base = bob::extension::VariableDoc(
  "jfa_base",
  ":py:class:`bob.learn.em.JFABase`",
  "The JFABase attached to this machine",
  ""
);
PyObject* PyBobLearnEMJFAMachine_getJFABase(PyBobLearnEMJFAMachineObject* self, void*){
  BOB_TRY

  boost::shared_ptr<bob::learn::em::JFABase> jfa_base_o = self->cxx->getJFABase();

  //Allocating the correspondent python object
  PyBobLearnEMJFABaseObject* retval =
    (PyBobLearnEMJFABaseObject*)PyBobLearnEMJFABase_Type.tp_alloc(&PyBobLearnEMJFABase_Type, 0);
  retval->cxx = jfa_base_o;

  return Py_BuildValue("N",retval);
  BOB_CATCH_MEMBER("jfa_base could not be read", 0)
}
int PyBobLearnEMJFAMachine_setJFABase(PyBobLearnEMJFAMachineObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBobLearnEMJFABase_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a :py:class:`bob.learn.em.JFABase`", Py_TYPE(self)->tp_name, jfa_base.name());
    return -1;
  }

  PyBobLearnEMJFABaseObject* jfa_base_o = 0;
  PyArg_Parse(value, "O!", &PyBobLearnEMJFABase_Type,&jfa_base_o);

  self->cxx->setJFABase(jfa_base_o->cxx);

  return 0;
  BOB_CATCH_MEMBER("jfa_base could not be set", -1)
}




static PyGetSetDef PyBobLearnEMJFAMachine_getseters[] = {
  {
   shape.name(),
   (getter)PyBobLearnEMJFAMachine_getShape,
   0,
   shape.doc(),
   0
  },

  {
   supervector_length.name(),
   (getter)PyBobLearnEMJFAMachine_getSupervectorLength,
   0,
   supervector_length.doc(),
   0
  },

  {
   jfa_base.name(),
   (getter)PyBobLearnEMJFAMachine_getJFABase,
   (setter)PyBobLearnEMJFAMachine_setJFABase,
   jfa_base.doc(),
   0
  },

  {
   Y.name(),
   (getter)PyBobLearnEMJFAMachine_getY,
   (setter)PyBobLearnEMJFAMachine_setY,
   Y.doc(),
   0
  },

  {
   Z.name(),
   (getter)PyBobLearnEMJFAMachine_getZ,
   (setter)PyBobLearnEMJFAMachine_setZ,
   Z.doc(),
   0
  },

  {
   X.name(),
   (getter)PyBobLearnEMJFAMachine_getX,
   0,
   X.doc(),
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
  "Save the configuration of the JFAMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnEMJFAMachine_Save(PyBobLearnEMJFAMachineObject* self,  PyObject* args, PyObject* kwargs) {

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
  "Load the configuration of the JFAMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnEMJFAMachine_Load(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {
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

  "Compares this JFAMachine with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.JFAMachine`", "A JFAMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMJFAMachine_IsSimilarTo(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMJFAMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMJFAMachine_Type, &other,
        &r_epsilon, &a_epsilon)){

        is_similar_to.print_usage();
        return 0;
  }

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}


/*** estimate_x ***/
static auto estimate_x = bob::extension::FunctionDoc(
  "estimate_x",
  "Estimates the session offset x (LPT assumption) given GMM statistics.",
  "Estimates x from the GMM statistics considering the LPT assumption, that is the latent session variable x is approximated using the UBM",
  true
)
.add_prototype("stats,input")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "Statistics of the GMM")
.add_parameter("input", "array_like <float, 1D>", "Input vector");
static PyObject* PyBobLearnEMJFAMachine_estimateX(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = estimate_x.kwlist(0);

  PyBobLearnEMGMMStatsObject* stats = 0;
  PyBlitzArrayObject* input           = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnEMGMMStats_Type, &stats,
                                                                 &PyBlitzArray_Converter,&input))
    return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, estimate_x.name());
    return 0;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, estimate_x.name());
    return 0;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()) {
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d, elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs(), (Py_ssize_t)input->shape[0], estimate_x.name());
    return 0;
  }

  self->cxx->estimateX(*stats->cxx, *PyBlitzArrayCxx_AsBlitz<double,1>(input));

  BOB_CATCH_MEMBER("cannot estimate X", 0)
  Py_RETURN_NONE;
}


/*** estimate_ux ***/
static auto estimate_ux = bob::extension::FunctionDoc(
  "estimate_ux",
  "Estimates Ux (LPT assumption) given GMM statistics.",
  "Estimates Ux from the GMM statistics considering the LPT assumption, that is the latent session variable x is approximated using the UBM.",
  true
)
.add_prototype("stats,input")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "Statistics of the GMM")
.add_parameter("input", "array_like <float, 1D>", "Input vector");
static PyObject* PyBobLearnEMJFAMachine_estimateUx(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = estimate_ux.kwlist(0);

  PyBobLearnEMGMMStatsObject* stats = 0;
  PyBlitzArrayObject* input           = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnEMGMMStats_Type, &stats,
                                                                 &PyBlitzArray_Converter,&input))
    return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);

  // perform check on the input
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, estimate_ux.name());
    return 0;
  }

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, estimate_ux.name());
    return 0;
  }

  if (input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()*(Py_ssize_t)self->cxx->getNInputs()) {
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d, elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, self->cxx->getNInputs()*(Py_ssize_t)self->cxx->getNGaussians(), (Py_ssize_t)input->shape[0], estimate_ux.name());
    return 0;
  }

  self->cxx->estimateUx(*stats->cxx, *PyBlitzArrayCxx_AsBlitz<double,1>(input));

  BOB_CATCH_MEMBER("cannot estimate Ux", 0)
  Py_RETURN_NONE;
}


/*** forward_ux ***/
static auto forward_ux = bob::extension::FunctionDoc(
  "forward_ux",
  "Computes a score for the given UBM statistics and given the Ux vector",
  "",
  true
)
.add_prototype("stats,ux")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "Statistics as input")
.add_parameter("ux", "array_like <float, 1D>", "Input vector");
static PyObject* PyBobLearnEMJFAMachine_ForwardUx(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = forward_ux.kwlist(0);

  PyBobLearnEMGMMStatsObject* stats = 0;
  PyBlitzArrayObject* ux_input        = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnEMGMMStats_Type, &stats,
                                                                 &PyBlitzArray_Converter,&ux_input))
    return 0;

  //protects acquired resources through this scope
  auto ux_input_ = make_safe(ux_input);

  // perform check on the input
  if (ux_input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, forward_ux.name());
    return 0;
  }

  if (ux_input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, forward_ux.name());
    return 0;
  }

  if (ux_input->shape[0] != (Py_ssize_t)self->cxx->getNGaussians()*(Py_ssize_t)self->cxx->getNInputs()) {
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d, elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getNGaussians()*(Py_ssize_t)self->cxx->getNInputs(), (Py_ssize_t)ux_input->shape[0], forward_ux.name());
    return 0;
  }

  double score = self->cxx->forward(*stats->cxx, *PyBlitzArrayCxx_AsBlitz<double,1>(ux_input));

  return Py_BuildValue("d", score);
  BOB_CATCH_MEMBER("cannot forward_ux", 0)

}


/*** log_likelihood ***/
static auto log_likelihood = bob::extension::FunctionDoc(
  "log_likelihood",
  "Computes the log-likelihood of the given samples",
  ".. note:: the ``__call__`` function is an alias for this function.",
  true
)
.add_prototype("stats")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "Statistics as input");
static PyObject* PyBobLearnEMJFAMachine_log_likelihood(PyBobLearnEMJFAMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = log_likelihood.kwlist(0);

  PyBobLearnEMGMMStatsObject* stats = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMGMMStats_Type, &stats))
    return 0;

  //protects acquired resources through this scope
  double score = self->cxx->forward(*stats->cxx);

  return Py_BuildValue("d", score);
  BOB_CATCH_MEMBER("cannot log_likelihood", 0)

}


static PyMethodDef PyBobLearnEMJFAMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnEMJFAMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMJFAMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMJFAMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },

  {
    estimate_x.name(),
    (PyCFunction)PyBobLearnEMJFAMachine_estimateX,
    METH_VARARGS|METH_KEYWORDS,
    estimate_x.doc()
  },

  {
    estimate_ux.name(),
    (PyCFunction)PyBobLearnEMJFAMachine_estimateUx,
    METH_VARARGS|METH_KEYWORDS,
    estimate_ux.doc()
  },

  {
    forward_ux.name(),
    (PyCFunction)PyBobLearnEMJFAMachine_ForwardUx,
    METH_VARARGS|METH_KEYWORDS,
    forward_ux.doc()
  },
  {
    log_likelihood.name(),
    (PyCFunction)PyBobLearnEMJFAMachine_log_likelihood,
    METH_VARARGS|METH_KEYWORDS,
    log_likelihood.doc()
  },


  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the JFA type struct; will be initialized later
PyTypeObject PyBobLearnEMJFAMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMJFAMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMJFAMachine_Type.tp_name      = JFAMachine_doc.name();
  PyBobLearnEMJFAMachine_Type.tp_basicsize = sizeof(PyBobLearnEMJFAMachineObject);
  PyBobLearnEMJFAMachine_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMJFAMachine_Type.tp_doc       = JFAMachine_doc.doc();

  // set the functions
  PyBobLearnEMJFAMachine_Type.tp_new         = PyType_GenericNew;
  PyBobLearnEMJFAMachine_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnEMJFAMachine_init);
  PyBobLearnEMJFAMachine_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnEMJFAMachine_delete);
  PyBobLearnEMJFAMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMJFAMachine_RichCompare);
  PyBobLearnEMJFAMachine_Type.tp_methods     = PyBobLearnEMJFAMachine_methods;
  PyBobLearnEMJFAMachine_Type.tp_getset      = PyBobLearnEMJFAMachine_getseters;
  PyBobLearnEMJFAMachine_Type.tp_call        = reinterpret_cast<ternaryfunc>(PyBobLearnEMJFAMachine_log_likelihood);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMJFAMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMJFAMachine_Type);
  return PyModule_AddObject(module, "JFAMachine", (PyObject*)&PyBobLearnEMJFAMachine_Type) >= 0;
}

/**
 * @date Wed Jan 28 13:03:15 2015 +0200
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

static auto ISVMachine_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".ISVMachine",
  "A ISVMachine. An attached :py:class:`bob.learn.em.ISVBase` should be provided for Joint Factor Analysis. The :py:class:`bob.learn.em.ISVMachine` carries information about the speaker factors :math:`y` and :math:`z`, whereas a :py:class:`bob.learn.em.JFABase` carries information about the matrices :math:`U` and :math:`D`.\n\n"
  "References: [Vogt2008]_ [McCool2013]_",
  ""
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructor. Builds a new ISVMachine",
    "",
    true
  )
  .add_prototype("isv_base","")
  .add_prototype("other","")
  .add_prototype("hdf5","")

  .add_parameter("isv_base", ":py:class:`bob.learn.em.ISVBase`", "The ISVBase associated with this machine")
  .add_parameter("other", ":py:class:`bob.learn.em.ISVMachine`", "A ISVMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnEMISVMachine_init_copy(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVMachine_doc.kwlist(1);
  PyBobLearnEMISVMachineObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMISVMachine_Type, &o)){
    ISVMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::ISVMachine(*o->cxx));
  return 0;
}


static int PyBobLearnEMISVMachine_init_hdf5(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    ISVMachine_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);
  self->cxx.reset(new bob::learn::em::ISVMachine(*(config->f)));

  return 0;
}


static int PyBobLearnEMISVMachine_init_isvbase(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVMachine_doc.kwlist(0);
  
  PyBobLearnEMISVBaseObject* isv_base;

  //Here we have to select which keyword argument to read  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMISVBase_Type, &isv_base)){
    ISVMachine_doc.print_usage();
    return -1;
  }
  
  self->cxx.reset(new bob::learn::em::ISVMachine(isv_base->cxx));
  return 0;
}


static int PyBobLearnEMISVMachine_init(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {
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
    if (PyBobLearnEMISVMachine_Check(arg))
      return PyBobLearnEMISVMachine_init_copy(self, args, kwargs);
    // If the constructor input is a HDF5
    else if (PyBobIoHDF5File_Check(arg))
      return PyBobLearnEMISVMachine_init_hdf5(self, args, kwargs);
    // If the constructor input is a JFABase Object
    else
      return PyBobLearnEMISVMachine_init_isvbase(self, args, kwargs);
  }
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires only 1 argument, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
    ISVMachine_doc.print_usage();
    return -1;
  }
  
  BOB_CATCH_MEMBER("cannot create ISVMachine", -1)
  return 0;
}

static void PyBobLearnEMISVMachine_delete(PyBobLearnEMISVMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMISVMachine_RichCompare(PyBobLearnEMISVMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMISVMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMISVMachineObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare ISVMachine objects", 0)
}

int PyBobLearnEMISVMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMISVMachine_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int, int, int)",
  "A tuple that represents the number of gaussians, dimensionality of each Gaussian and dimensionality of the rU (within client variability matrix)) ``(#Gaussians, #Inputs, #rU)``.",
  ""
);
PyObject* PyBobLearnEMISVMachine_getShape(PyBobLearnEMISVMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i,i)", self->cxx->getNGaussians(), self->cxx->getNInputs(), self->cxx->getDimRu());
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
PyObject* PyBobLearnEMISVMachine_getSupervectorLength(PyBobLearnEMISVMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getSupervectorLength());
  BOB_CATCH_MEMBER("supervector_length could not be read", 0)
}

/***** z *****/
static auto Z = bob::extension::VariableDoc(
  "z",
  "array_like <float, 1D>",
  "Returns the :math:`z` speaker factor. Eq (31) from [McCool2013]_",
  ""
);
PyObject* PyBobLearnEMISVMachine_getZ(PyBobLearnEMISVMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZ());
  BOB_CATCH_MEMBER("`z` could not be read", 0)
}
int PyBobLearnEMISVMachine_setZ(PyBobLearnEMISVMachineObject* self, PyObject* value, void*){
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
  "The latent variable x (last one computed). This is a feature provided for convenience, but this attribute is not 'part' of the machine. The session latent variable :math:`x` is indeed not class-specific, but depends on the sample considered. Furthermore, it is not saved into the machine or used when comparing machines."
);
PyObject* PyBobLearnEMISVMachine_getX(PyBobLearnEMISVMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getX());
  BOB_CATCH_MEMBER("`x` could not be read", 0)
}


/***** isv_base *****/
static auto isv_base = bob::extension::VariableDoc(
  "isv_base",
  ":py:class:`bob.learn.em.ISVBase`",
  "The ISVBase attached to this machine",
  ""
);
PyObject* PyBobLearnEMISVMachine_getISVBase(PyBobLearnEMISVMachineObject* self, void*){
  BOB_TRY

  boost::shared_ptr<bob::learn::em::ISVBase> isv_base_o = self->cxx->getISVBase();

  //Allocating the correspondent python object
  PyBobLearnEMISVBaseObject* retval =
    (PyBobLearnEMISVBaseObject*)PyBobLearnEMISVBase_Type.tp_alloc(&PyBobLearnEMISVBase_Type, 0);
  retval->cxx = isv_base_o;

  return Py_BuildValue("N",retval);
  BOB_CATCH_MEMBER("isv_base could not be read", 0)
}
int PyBobLearnEMISVMachine_setISVBase(PyBobLearnEMISVMachineObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBobLearnEMISVBase_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a :py:class:`bob.learn.em.ISVBase`", Py_TYPE(self)->tp_name, isv_base.name());
    return -1;
  }

  PyBobLearnEMISVBaseObject* isv_base_o = 0;
  PyArg_Parse(value, "O!", &PyBobLearnEMISVBase_Type,&isv_base_o);

  self->cxx->setISVBase(isv_base_o->cxx);

  return 0;
  BOB_CATCH_MEMBER("isv_base could not be set", -1)  
}




static PyGetSetDef PyBobLearnEMISVMachine_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnEMISVMachine_getShape,
   0,
   shape.doc(),
   0
  },
  
  {
   supervector_length.name(),
   (getter)PyBobLearnEMISVMachine_getSupervectorLength,
   0,
   supervector_length.doc(),
   0
  },
  
  {
   isv_base.name(),
   (getter)PyBobLearnEMISVMachine_getISVBase,
   (setter)PyBobLearnEMISVMachine_setISVBase,
   isv_base.doc(),
   0
  },

  {
   Z.name(),
   (getter)PyBobLearnEMISVMachine_getZ,
   (setter)PyBobLearnEMISVMachine_setZ,
   Z.doc(),
   0
  },

  {
   X.name(),
   (getter)PyBobLearnEMISVMachine_getX,
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
  "Save the configuration of the ISVMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnEMISVMachine_Save(PyBobLearnEMISVMachineObject* self,  PyObject* args, PyObject* kwargs) {

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
  "Load the configuration of the ISVMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnEMISVMachine_Load(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {
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
  
  "Compares this ISVMachine with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.ISVMachine`", "A ISVMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMISVMachine_IsSimilarTo(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMISVMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMISVMachine_Type, &other,
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
  "Estimates :math:`x` from the GMM statistics considering the LPT assumption, that is the latent session variable :math:`x` is approximated using the UBM", 
  true
)
.add_prototype("stats,input")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "Statistics of the GMM")
.add_parameter("input", "array_like <float, 1D>", "Input vector");
static PyObject* PyBobLearnEMISVMachine_estimateX(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {
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
static PyObject* PyBobLearnEMISVMachine_estimateUx(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {
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
static PyObject* PyBobLearnEMISVMachine_ForwardUx(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {
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


/*** forward ***/
static auto forward = bob::extension::FunctionDoc(
  "forward",
  "Execute the machine",
  "", 
  true
)
.add_prototype("stats")
.add_parameter("stats", ":py:class:`bob.learn.em.GMMStats`", "Statistics as input");
static PyObject* PyBobLearnEMISVMachine_Forward(PyBobLearnEMISVMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = forward.kwlist(0);

  PyBobLearnEMGMMStatsObject* stats = 0;
  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMGMMStats_Type, &stats))
    return 0;

  //protects acquired resources through this scope
  double score = self->cxx->forward(*stats->cxx);

  return Py_BuildValue("d", score);
  BOB_CATCH_MEMBER("cannot forward", 0)

}


static PyMethodDef PyBobLearnEMISVMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnEMISVMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMISVMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMISVMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  
  {
    estimate_x.name(),
    (PyCFunction)PyBobLearnEMISVMachine_estimateX,
    METH_VARARGS|METH_KEYWORDS,
    estimate_x.doc()
  },
  
  {
    estimate_ux.name(),
    (PyCFunction)PyBobLearnEMISVMachine_estimateUx,
    METH_VARARGS|METH_KEYWORDS,
    estimate_ux.doc()
  },

  {
    forward_ux.name(),
    (PyCFunction)PyBobLearnEMISVMachine_ForwardUx,
    METH_VARARGS|METH_KEYWORDS,
    forward_ux.doc()
  },

  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the JFA type struct; will be initialized later
PyTypeObject PyBobLearnEMISVMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMISVMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMISVMachine_Type.tp_name      = ISVMachine_doc.name();
  PyBobLearnEMISVMachine_Type.tp_basicsize = sizeof(PyBobLearnEMISVMachineObject);
  PyBobLearnEMISVMachine_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMISVMachine_Type.tp_doc       = ISVMachine_doc.doc();

  // set the functions
  PyBobLearnEMISVMachine_Type.tp_new         = PyType_GenericNew;
  PyBobLearnEMISVMachine_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnEMISVMachine_init);
  PyBobLearnEMISVMachine_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnEMISVMachine_delete);
  PyBobLearnEMISVMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMISVMachine_RichCompare);
  PyBobLearnEMISVMachine_Type.tp_methods     = PyBobLearnEMISVMachine_methods;
  PyBobLearnEMISVMachine_Type.tp_getset      = PyBobLearnEMISVMachine_getseters;
  PyBobLearnEMISVMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMISVMachine_Forward);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMISVMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMISVMachine_Type);
  return PyModule_AddObject(module, "ISVMachine", (PyObject*)&PyBobLearnEMISVMachine_Type) >= 0;
}


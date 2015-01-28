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
  "A ISVMachine. An attached :py:class:`bob.learn.misc.ISVBase` should be provided for Joint Factor Analysis. The :py:class:`bob.learn.misc.ISVMachine` carries information about the speaker factors y and z, whereas a :py:class:`bob.learn.misc.JFABase` carries information about the matrices U, V and D."
  "References: [Vogt2008,McCool2013]",
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

  .add_parameter("isv", ":py:class:`bob.learn.misc.ISVBase`", "The ISVBase associated with this machine")
  .add_parameter("other", ":py:class:`bob.learn.misc.ISVMachine`", "A ISVMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnMiscISVMachine_init_copy(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVMachine_doc.kwlist(1);
  PyBobLearnMiscISVMachineObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscISVMachine_Type, &o)){
    ISVMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::ISVMachine(*o->cxx));
  return 0;
}


static int PyBobLearnMiscISVMachine_init_hdf5(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    ISVMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::ISVMachine(*(config->f)));

  return 0;
}


static int PyBobLearnMiscISVMachine_init_isvbase(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVMachine_doc.kwlist(0);
  
  PyBobLearnMiscISVBaseObject* isv_base;

  //Here we have to select which keyword argument to read  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscISVBase_Type, &isv_base)){
    ISVMachine_doc.print_usage();
    return -1;
  }
  
  self->cxx.reset(new bob::learn::misc::ISVMachine(isv_base->cxx));
  return 0;
}


static int PyBobLearnMiscISVMachine_init(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {
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
    if (PyBobLearnMiscISVMachine_Check(arg))
      return PyBobLearnMiscISVMachine_init_copy(self, args, kwargs);
    // If the constructor input is a HDF5
    else if (PyBobIoHDF5File_Check(arg))
      return PyBobLearnMiscISVMachine_init_hdf5(self, args, kwargs);
    // If the constructor input is a JFABase Object
    else
      return PyBobLearnMiscISVMachine_init_isvbase(self, args, kwargs);
  }
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires only 1 argument, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
    ISVMachine_doc.print_usage();
    return -1;
  }
  
  BOB_CATCH_MEMBER("cannot create ISVMachine", 0)
  return 0;
}

static void PyBobLearnMiscISVMachine_delete(PyBobLearnMiscISVMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnMiscISVMachine_RichCompare(PyBobLearnMiscISVMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscISVMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscISVMachineObject*>(other);
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

int PyBobLearnMiscISVMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscISVMachine_Type));
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
PyObject* PyBobLearnMiscISVMachine_getShape(PyBobLearnMiscISVMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i,i)", self->cxx->getNGaussians(), self->cxx->getNInputs(), self->cxx->getDimRu());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}

/***** supervector_length *****/
static auto supervector_length = bob::extension::VariableDoc(
  "supervector_length",
  "int",

  "Returns the supervector length."
  "NGaussians x NInputs: Number of Gaussian components by the feature dimensionality",
  
  "@warning An exception is thrown if no Universal Background Model has been set yet."
);
PyObject* PyBobLearnMiscISVMachine_getSupervectorLength(PyBobLearnMiscISVMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getSupervectorLength());
  BOB_CATCH_MEMBER("supervector_length could not be read", 0)
}

/***** z *****/
static auto Z = bob::extension::VariableDoc(
  "z",
  "array_like <float, 1D>",
  "Returns the z speaker factor. Eq (31) from [McCool2013]",
  ""
);
PyObject* PyBobLearnMiscISVMachine_getZ(PyBobLearnMiscISVMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getZ());
  BOB_CATCH_MEMBER("`z` could not be read", 0)
}
int PyBobLearnMiscISVMachine_setZ(PyBobLearnMiscISVMachineObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, Z.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "z");
  if (!b) return -1;
  self->cxx->setZ(*b);
  return 0;
  BOB_CATCH_MEMBER("`z` vector could not be set", -1)
}


/***** x *****/
static auto X = bob::extension::VariableDoc(
  "x",
  "array_like <float, 1D>",
  "Returns the X session factor. Eq (29) from [McCool2013]",
  "The latent variable x (last one computed). This is a feature provided for convenience, but this attribute is not 'part' of the machine. The session latent variable x is indeed not class-specific, but depends on the sample considered. Furthermore, it is not saved into the machine or used when comparing machines."
);
PyObject* PyBobLearnMiscISVMachine_getX(PyBobLearnMiscISVMachineObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getX());
  BOB_CATCH_MEMBER("`x` could not be read", 0)
}


/***** isv_base *****/
static auto isv_base = bob::extension::VariableDoc(
  "isv_base",
  ":py:class:`bob.learn.misc.ISVBase`",
  "The ISVBase attached to this machine",
  ""
);
PyObject* PyBobLearnMiscISVMachine_getISVBase(PyBobLearnMiscISVMachineObject* self, void*){
  BOB_TRY

  boost::shared_ptr<bob::learn::misc::ISVBase> isv_base_o = self->cxx->getISVBase();

  //Allocating the correspondent python object
  PyBobLearnMiscISVBaseObject* retval =
    (PyBobLearnMiscISVBaseObject*)PyBobLearnMiscISVBase_Type.tp_alloc(&PyBobLearnMiscISVBase_Type, 0);
  retval->cxx = isv_base_o;

  return Py_BuildValue("O",retval);
  BOB_CATCH_MEMBER("isv_base could not be read", 0)
}
int PyBobLearnMiscISVMachine_setISVBase(PyBobLearnMiscISVMachineObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBobLearnMiscISVBase_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a :py:class:`bob.learn.misc.ISVBase`", Py_TYPE(self)->tp_name, isv_base.name());
    return -1;
  }

  PyBobLearnMiscISVBaseObject* isv_base_o = 0;
  PyArg_Parse(value, "O!", &PyBobLearnMiscISVBase_Type,&isv_base_o);

  self->cxx->setISVBase(isv_base_o->cxx);

  return 0;
  BOB_CATCH_MEMBER("isv_base could not be set", -1)  
}




static PyGetSetDef PyBobLearnMiscISVMachine_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnMiscISVMachine_getShape,
   0,
   shape.doc(),
   0
  },
  
  {
   supervector_length.name(),
   (getter)PyBobLearnMiscISVMachine_getSupervectorLength,
   0,
   supervector_length.doc(),
   0
  },
  
  {
   isv_base.name(),
   (getter)PyBobLearnMiscISVMachine_getISVBase,
   (setter)PyBobLearnMiscISVMachine_setISVBase,
   isv_base.doc(),
   0
  },

  {
   Z.name(),
   (getter)PyBobLearnMiscISVMachine_getZ,
   (setter)PyBobLearnMiscISVMachine_setZ,
   Z.doc(),
   0
  },

  {
   X.name(),
   (getter)PyBobLearnMiscISVMachine_getX,
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
static PyObject* PyBobLearnMiscISVMachine_Save(PyBobLearnMiscISVMachineObject* self,  PyObject* args, PyObject* kwargs) {

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
static PyObject* PyBobLearnMiscISVMachine_Load(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {
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
.add_parameter("other", ":py:class:`bob.learn.misc.ISVMachine`", "A ISVMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnMiscISVMachine_IsSimilarTo(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnMiscISVMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnMiscISVMachine_Type, &other,
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
.add_parameter("stats", ":py:class:`bob.learn.misc.GMMStats`", "Statistics of the GMM")
.add_parameter("input", "array_like <float, 1D>", "Input vector");
static PyObject* PyBobLearnMiscISVMachine_estimateX(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = estimate_x.kwlist(0);

  PyBobLearnMiscGMMStatsObject* stats = 0;
  PyBlitzArrayObject* input           = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnMiscGMMStats_Type, &stats, 
                                                                 &PyBlitzArray_Converter,&input))
    Py_RETURN_NONE;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
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
.add_parameter("stats", ":py:class:`bob.learn.misc.GMMStats`", "Statistics of the GMM")
.add_parameter("input", "array_like <float, 1D>", "Input vector");
static PyObject* PyBobLearnMiscISVMachine_estimateUx(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = estimate_ux.kwlist(0);

  PyBobLearnMiscGMMStatsObject* stats = 0;
  PyBlitzArrayObject* input           = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnMiscGMMStats_Type, &stats, 
                                                                 &PyBlitzArray_Converter,&input))
    Py_RETURN_NONE;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
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
.add_parameter("stats", ":py:class:`bob.learn.misc.GMMStats`", "Statistics as input")
.add_parameter("ux", "array_like <float, 1D>", "Input vector");
static PyObject* PyBobLearnMiscISVMachine_ForwardUx(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = forward_ux.kwlist(0);

  PyBobLearnMiscGMMStatsObject* stats = 0;
  PyBlitzArrayObject* ux_input        = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&", kwlist, &PyBobLearnMiscGMMStats_Type, &stats, 
                                                                 &PyBlitzArray_Converter,&ux_input))
    Py_RETURN_NONE;

  //protects acquired resources through this scope
  auto ux_input_ = make_safe(ux_input);
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
.add_parameter("stats", ":py:class:`bob.learn.misc.GMMStats`", "Statistics as input");
static PyObject* PyBobLearnMiscISVMachine_Forward(PyBobLearnMiscISVMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist = forward.kwlist(0);

  PyBobLearnMiscGMMStatsObject* stats = 0;
  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMStats_Type, &stats))
    Py_RETURN_NONE;

  //protects acquired resources through this scope
  double score = self->cxx->forward(*stats->cxx);

  return Py_BuildValue("d", score);
  BOB_CATCH_MEMBER("cannot forward", 0)

}


static PyMethodDef PyBobLearnMiscISVMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnMiscISVMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnMiscISVMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnMiscISVMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  
  {
    estimate_x.name(),
    (PyCFunction)PyBobLearnMiscISVMachine_estimateX,
    METH_VARARGS|METH_KEYWORDS,
    estimate_x.doc()
  },
  
  {
    estimate_ux.name(),
    (PyCFunction)PyBobLearnMiscISVMachine_estimateUx,
    METH_VARARGS|METH_KEYWORDS,
    estimate_ux.doc()
  },

  {
    forward_ux.name(),
    (PyCFunction)PyBobLearnMiscISVMachine_ForwardUx,
    METH_VARARGS|METH_KEYWORDS,
    forward_ux.doc()
  },

  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the JFA type struct; will be initialized later
PyTypeObject PyBobLearnMiscISVMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscISVMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscISVMachine_Type.tp_name      = ISVMachine_doc.name();
  PyBobLearnMiscISVMachine_Type.tp_basicsize = sizeof(PyBobLearnMiscISVMachineObject);
  PyBobLearnMiscISVMachine_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscISVMachine_Type.tp_doc       = ISVMachine_doc.doc();

  // set the functions
  PyBobLearnMiscISVMachine_Type.tp_new         = PyType_GenericNew;
  PyBobLearnMiscISVMachine_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnMiscISVMachine_init);
  PyBobLearnMiscISVMachine_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnMiscISVMachine_delete);
  PyBobLearnMiscISVMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscISVMachine_RichCompare);
  PyBobLearnMiscISVMachine_Type.tp_methods     = PyBobLearnMiscISVMachine_methods;
  PyBobLearnMiscISVMachine_Type.tp_getset      = PyBobLearnMiscISVMachine_getseters;
  PyBobLearnMiscISVMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnMiscISVMachine_Forward);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscISVMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscISVMachine_Type);
  return PyModule_AddObject(module, "ISVMachine", (PyObject*)&PyBobLearnMiscISVMachine_Type) >= 0;
}


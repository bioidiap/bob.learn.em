/**
 * @date Tue Jan 27 17:03:15 2015 +0200
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

static auto JFABase_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".JFABase",

  "Constructor. Builds a new JFABase. "
  "TODO: add a reference to the journal articles",
  ""
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a FABase",
    "",
    true
  )
  .add_prototype("gmm,ru,rv","")
  .add_prototype("other","")
  .add_prototype("hdf5","")
  .add_prototype("","")

  .add_parameter("gmm", ":py:class:`bob.learn.misc.GMMMachine`", "The Universal Background Model.")
  .add_parameter("ru", "int", "Size of U (Within client variation matrix). In the end the U matrix will have (number_of_gaussians * feature_dimension x ru)")
  .add_parameter("rv", "int", "Size of V (Between client variation matrix). In the end the U matrix will have (number_of_gaussians * feature_dimension x rv)")
  .add_parameter("other", ":py:class:`bob.learn.misc.JFABase`", "A JFABase object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnMiscJFABase_init_copy(PyBobLearnMiscJFABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFABase_doc.kwlist(1);
  PyBobLearnMiscJFABaseObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscJFABase_Type, &o)){
    JFABase_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::JFABase(*o->cxx));
  return 0;
}


static int PyBobLearnMiscJFABase_init_hdf5(PyBobLearnMiscJFABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFABase_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    JFABase_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::JFABase(*(config->f)));

  return 0;
}


static int PyBobLearnMiscJFABase_init_ubm(PyBobLearnMiscJFABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFABase_doc.kwlist(0);
  
  PyBobLearnMiscGMMMachineObject* ubm;
  int ru = 1;
  int rv = 1;

  //Here we have to select which keyword argument to read  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ii", kwlist, &PyBobLearnMiscGMMMachine_Type, &ubm,
                                                                &ru, &rv)){
    JFABase_doc.print_usage();
    return -1;
  }
  
  self->cxx.reset(new bob::learn::misc::JFABase(ubm->cxx, ru, rv));
  return 0;
}


static int PyBobLearnMiscJFABase_init(PyBobLearnMiscJFABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);
    
  switch (nargs) {

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
     if (PyBobLearnMiscJFABase_Check(arg))
       return PyBobLearnMiscJFABase_init_copy(self, args, kwargs);
      // If the constructor input is a HDF5
     else if (PyBobIoHDF5File_Check(arg))
       return PyBobLearnMiscJFABase_init_hdf5(self, args, kwargs);
    }
    case 3:
      return PyBobLearnMiscJFABase_init_ubm(self, args, kwargs);
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 3 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      JFABase_doc.print_usage();
      return -1;
  }
  BOB_CATCH_MEMBER("cannot create JFABase", 0)
  return 0;
}



static void PyBobLearnMiscJFABase_delete(PyBobLearnMiscJFABaseObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnMiscJFABase_RichCompare(PyBobLearnMiscJFABaseObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscJFABase_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscJFABaseObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare JFABase objects", 0)
}

int PyBobLearnMiscJFABase_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscJFABase_Type));
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
PyObject* PyBobLearnMiscJFABase_getShape(PyBobLearnMiscJFABaseObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i,i,i)", self->cxx->getNGaussians(), self->cxx->getNInputs(), self->cxx->getDimRu(), self->cxx->getDimRv());
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
PyObject* PyBobLearnMiscJFABase_getSupervectorLength(PyBobLearnMiscJFABaseObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getSupervectorLength());
  BOB_CATCH_MEMBER("supervector_length could not be read", 0)
}


/***** u *****/
static auto U = bob::extension::VariableDoc(
  "u",
  "array_like <float, 2D>",
  "Returns the U matrix (within client variability matrix)",
  ""
);
PyObject* PyBobLearnMiscJFABase_getU(PyBobLearnMiscJFABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getU());
  BOB_CATCH_MEMBER("``u`` could not be read", 0)
}
int PyBobLearnMiscJFABase_setU(PyBobLearnMiscJFABaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, U.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(o, "u");
  if (!b) return -1;
  self->cxx->setU(*b);
  return 0;
  BOB_CATCH_MEMBER("``u`` matrix could not be set", -1)
}

/***** v *****/
static auto V = bob::extension::VariableDoc(
  "v",
  "array_like <float, 2D>",
  "Returns the V matrix (between client variability matrix)",
  ""
);
PyObject* PyBobLearnMiscJFABase_getV(PyBobLearnMiscJFABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getV());
  BOB_CATCH_MEMBER("``v`` could not be read", 0)
}
int PyBobLearnMiscJFABase_setV(PyBobLearnMiscJFABaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, V.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(o, "v");
  if (!b) return -1;
  self->cxx->setV(*b);
  return 0;
  BOB_CATCH_MEMBER("``v`` matrix could not be set", -1)
}


/***** d *****/
static auto D = bob::extension::VariableDoc(
  "d",
  "array_like <float, 1D>",
  "Returns the diagonal matrix diag(d) (as a 1D vector)",
  ""
);
PyObject* PyBobLearnMiscJFABase_getD(PyBobLearnMiscJFABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getD());
  BOB_CATCH_MEMBER("``d`` could not be read", 0)
}
int PyBobLearnMiscJFABase_setD(PyBobLearnMiscJFABaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, D.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "d");
  if (!b) return -1;
  self->cxx->setD(*b);
  return 0;
  BOB_CATCH_MEMBER("``d`` matrix could not be set", -1)
}


/***** ubm *****/
static auto ubm = bob::extension::VariableDoc(
  "ubm",
  ":py:class:`bob.learn.misc.GMMMachine`",
  "Returns the UBM (Universal Background Model",
  ""
);
PyObject* PyBobLearnMiscJFABase_getUBM(PyBobLearnMiscJFABaseObject* self, void*){
  BOB_TRY

  boost::shared_ptr<bob::learn::misc::GMMMachine> ubm_gmmMachine = self->cxx->getUbm();

  //Allocating the correspondent python object
  PyBobLearnMiscGMMMachineObject* retval =
    (PyBobLearnMiscGMMMachineObject*)PyBobLearnMiscGMMMachine_Type.tp_alloc(&PyBobLearnMiscGMMMachine_Type, 0);
  retval->cxx = ubm_gmmMachine;

  return Py_BuildValue("O",retval);
  BOB_CATCH_MEMBER("ubm could not be read", 0)
}
int PyBobLearnMiscJFABase_setUBM(PyBobLearnMiscJFABaseObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBobLearnMiscGMMMachine_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a :py:class:`bob.learn.misc.GMMMachine`", Py_TYPE(self)->tp_name, ubm.name());
    return -1;
  }

  PyBobLearnMiscGMMMachineObject* ubm_gmmMachine = 0;
  PyArg_Parse(value, "O!", &PyBobLearnMiscGMMMachine_Type,&ubm_gmmMachine);

  self->cxx->setUbm(ubm_gmmMachine->cxx);

  return 0;
  BOB_CATCH_MEMBER("ubm could not be set", -1)  
}




static PyGetSetDef PyBobLearnMiscJFABase_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnMiscJFABase_getShape,
   0,
   shape.doc(),
   0
  },
  
  {
   supervector_length.name(),
   (getter)PyBobLearnMiscJFABase_getSupervectorLength,
   0,
   supervector_length.doc(),
   0
  },
  
  {
   U.name(),
   (getter)PyBobLearnMiscJFABase_getU,
   (setter)PyBobLearnMiscJFABase_setU,
   U.doc(),
   0
  },
  
  {
   V.name(),
   (getter)PyBobLearnMiscJFABase_getV,
   (setter)PyBobLearnMiscJFABase_setV,
   V.doc(),
   0
  },

  {
   D.name(),
   (getter)PyBobLearnMiscJFABase_getD,
   (setter)PyBobLearnMiscJFABase_setD,
   D.doc(),
   0
  },

  {
   ubm.name(),
   (getter)PyBobLearnMiscJFABase_getUBM,
   (setter)PyBobLearnMiscJFABase_setUBM,
   ubm.doc(),
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
  "Save the configuration of the JFABase to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnMiscJFABase_Save(PyBobLearnMiscJFABaseObject* self,  PyObject* args, PyObject* kwargs) {

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
  "Load the configuration of the JFABase to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnMiscJFABase_Load(PyBobLearnMiscJFABaseObject* self, PyObject* args, PyObject* kwargs) {
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
  
  "Compares this JFABase with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.misc.JFABase`", "A JFABase object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnMiscJFABase_IsSimilarTo(PyBobLearnMiscJFABaseObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnMiscJFABaseObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnMiscJFABase_Type, &other,
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
static PyObject* PyBobLearnMiscJFABase_resize(PyBobLearnMiscJFABaseObject* self, PyObject* args, PyObject* kwargs) {
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




static PyMethodDef PyBobLearnMiscJFABase_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnMiscJFABase_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnMiscJFABase_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnMiscJFABase_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnMiscJFABase_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the JFA type struct; will be initialized later
PyTypeObject PyBobLearnMiscJFABase_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscJFABase(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscJFABase_Type.tp_name      = JFABase_doc.name();
  PyBobLearnMiscJFABase_Type.tp_basicsize = sizeof(PyBobLearnMiscJFABaseObject);
  PyBobLearnMiscJFABase_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscJFABase_Type.tp_doc       = JFABase_doc.doc();

  // set the functions
  PyBobLearnMiscJFABase_Type.tp_new         = PyType_GenericNew;
  PyBobLearnMiscJFABase_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnMiscJFABase_init);
  PyBobLearnMiscJFABase_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnMiscJFABase_delete);
  PyBobLearnMiscJFABase_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscJFABase_RichCompare);
  PyBobLearnMiscJFABase_Type.tp_methods     = PyBobLearnMiscJFABase_methods;
  PyBobLearnMiscJFABase_Type.tp_getset      = PyBobLearnMiscJFABase_getseters;
  //PyBobLearnMiscJFABase_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnMiscJFABase_forward);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscJFABase_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscJFABase_Type);
  return PyModule_AddObject(module, "JFABase", (PyObject*)&PyBobLearnMiscJFABase_Type) >= 0;
}


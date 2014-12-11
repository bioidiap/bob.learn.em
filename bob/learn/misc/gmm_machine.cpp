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
  .add_parameter("other", ":py:class:`bob.learn.misc.GMMMachine`", "A GMMMachine object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnMiscGMMMachine_init_number(PyBobLearnMiscGMMMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMMachine_doc.kwlist(0);
  int n_inputs    = 1;
  int n_gaussians = 1;
  //Parsing the input argments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &n_gaussians, &n_inputs))
    return -1;

  if(n_gaussians < 0){
    PyErr_Format(PyExc_TypeError, "gaussians argument must be greater than or equal to zero");
    GMMMachine_doc.print_usage();
    return -1;
  }

  if(n_inputs < 0){
    PyErr_Format(PyExc_TypeError, "input argument must be greater than or equal to zero");
    GMMMachine_doc.print_usage();
    return -1;
   }

  self->cxx.reset(new bob::learn::misc::GMMMachine(n_gaussians, n_inputs));
  return 0;
}


static int PyBobLearnMiscGMMMachine_init_copy(PyBobLearnMiscGMMMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMMachine_doc.kwlist(1);
  PyBobLearnMiscGMMMachineObject* tt;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscGMMMachine_Type, &tt)){
    GMMMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::GMMMachine(*tt->cxx));
  return 0;
}


static int PyBobLearnMiscGMMMachine_init_hdf5(PyBobLearnMiscGMMMachineObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = GMMMachine_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    GMMMachine_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::GMMMachine(*(config->f)));

  return 0;
}



static int PyBobLearnMiscGMMMachine_init(PyBobLearnMiscGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);
  
  switch (nargs) {

    case 0: //default initializer ()
      self->cxx.reset(new bob::learn::misc::GMMMachine());
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
     if (PyBobLearnMiscGMMMachine_Check(arg))
       return PyBobLearnMiscGMMMachine_init_copy(self, args, kwargs);
      // If the constructor input is a HDF5
     else if (PyBobIoHDF5File_Check(arg))
       return PyBobLearnMiscGMMMachine_init_hdf5(self, args, kwargs);
    }
    case 2:
      return PyBobLearnMiscGMMMachine_init_number(self, args, kwargs);
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0, 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      GMMMachine_doc.print_usage();
      return -1;
  }
  BOB_CATCH_MEMBER("cannot create GMMMachine", 0)
  return 0;
}



static void PyBobLearnMiscGMMMachine_delete(PyBobLearnMiscGMMMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnMiscGMMMachine_RichCompare(PyBobLearnMiscGMMMachineObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscGMMMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscGMMMachineObject*>(other);
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

int PyBobLearnMiscGMMMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscGMMMachine_Type));
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
PyObject* PyBobLearnMiscGMMMachine_getShape(PyBobLearnMiscGMMMachineObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i)", self->cxx->getNGaussians(), self->cxx->getNInputs());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}



static PyGetSetDef PyBobLearnMiscGMMMachine_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnMiscGMMMachine_getShape,
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
  "Save the configuration of the GMMMachine to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnMiscGMMMachine_Save(PyBobLearnMiscGMMMachineObject* self,  PyObject* args, PyObject* kwargs) {

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
static PyObject* PyBobLearnMiscGMMMachine_Load(PyBobLearnMiscGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
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
.add_parameter("other", ":py:class:`bob.learn.misc.GMMMachine`", "A GMMMachine object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnMiscGMMMachine_IsSimilarTo(PyBobLearnMiscGMMMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnMiscGMMMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnMiscGMMMachine_Type, &other,
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
static PyObject* PyBobLearnMiscGMMMachine_resize(PyBobLearnMiscGMMMachineObject* self, PyObject* args, PyObject* kwargs) {
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



static PyMethodDef PyBobLearnMiscGMMMachine_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnMiscGMMMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnMiscGMMMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnMiscGMMMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnMiscGMMMachine_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscGMMMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscGMMMachine(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscGMMMachine_Type.tp_name = GMMMachine_doc.name();
  PyBobLearnMiscGMMMachine_Type.tp_basicsize = sizeof(PyBobLearnMiscGMMMachineObject);
  PyBobLearnMiscGMMMachine_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscGMMMachine_Type.tp_doc = GMMMachine_doc.doc();

  // set the functions
  PyBobLearnMiscGMMMachine_Type.tp_new = PyType_GenericNew;
  PyBobLearnMiscGMMMachine_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnMiscGMMMachine_init);
  PyBobLearnMiscGMMMachine_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnMiscGMMMachine_delete);
  PyBobLearnMiscGMMMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscGMMMachine_RichCompare);
  PyBobLearnMiscGMMMachine_Type.tp_methods = PyBobLearnMiscGMMMachine_methods;
  PyBobLearnMiscGMMMachine_Type.tp_getset = PyBobLearnMiscGMMMachine_getseters;
  PyBobLearnMiscGMMMachine_Type.tp_call = 0;


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscGMMMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscGMMMachine_Type);
  return PyModule_AddObject(module, "GMMMachine", (PyObject*)&PyBobLearnMiscGMMMachine_Type) >= 0;
}


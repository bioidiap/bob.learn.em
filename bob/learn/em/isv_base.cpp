/**
 * @date Wed Jan 28 11:13:15 2015 +0200
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

static auto ISVBase_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".ISVBase",

  "A ISVBase instance can be seen as a container for U and D when performing Joint Factor Analysis (JFA).\n\n"
  "References: [Vogt2008]_ [McCool2013]_",
  ""
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a ISVBase",
    "",
    true
  )
  .add_prototype("ubm,ru","")
  .add_prototype("other","")
  .add_prototype("hdf5","")
  .add_prototype("","")

  .add_parameter("ubm", ":py:class:`bob.learn.em.GMMMachine`", "The Universal Background Model.")
  .add_parameter("ru", "int", "Size of U (Within client variation matrix). In the end the U matrix will have (number_of_gaussians * feature_dimension x ru)")
  .add_parameter("other", ":py:class:`bob.learn.em.ISVBase`", "A ISVBase object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnEMISVBase_init_copy(PyBobLearnEMISVBaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVBase_doc.kwlist(1);
  PyBobLearnEMISVBaseObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMISVBase_Type, &o)){
    ISVBase_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::ISVBase(*o->cxx));
  return 0;
}


static int PyBobLearnEMISVBase_init_hdf5(PyBobLearnEMISVBaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVBase_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    ISVBase_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);

  self->cxx.reset(new bob::learn::em::ISVBase(*(config->f)));

  return 0;
}


static int PyBobLearnEMISVBase_init_ubm(PyBobLearnEMISVBaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = ISVBase_doc.kwlist(0);
  
  PyBobLearnEMGMMMachineObject* ubm;
  int ru = 1;

  //Here we have to select which keyword argument to read  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!i", kwlist, &PyBobLearnEMGMMMachine_Type, &ubm, &ru)){
    ISVBase_doc.print_usage();
    return -1;
  }
  
  if(ru < 0){
    PyErr_Format(PyExc_TypeError, "ru argument must be greater than or equal to one");
    return -1;
  }
  
  self->cxx.reset(new bob::learn::em::ISVBase(ubm->cxx, ru));
  return 0;
}


static int PyBobLearnEMISVBase_init(PyBobLearnEMISVBaseObject* self, PyObject* args, PyObject* kwargs) {
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
     if (PyBobLearnEMISVBase_Check(arg))
       return PyBobLearnEMISVBase_init_copy(self, args, kwargs);
      // If the constructor input is a HDF5
     else if (PyBobIoHDF5File_Check(arg))
       return PyBobLearnEMISVBase_init_hdf5(self, args, kwargs);
    }
    case 2:
      return PyBobLearnEMISVBase_init_ubm(self, args, kwargs);
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 2 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      ISVBase_doc.print_usage();
      return -1;
  }
  BOB_CATCH_MEMBER("cannot create ISVBase", -1)
  return 0;
}


static void PyBobLearnEMISVBase_delete(PyBobLearnEMISVBaseObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMISVBase_RichCompare(PyBobLearnEMISVBaseObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMISVBase_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMISVBaseObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare ISVBase objects", 0)
}

int PyBobLearnEMISVBase_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMISVBase_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int, int)",
  "A tuple that represents the number of gaussians, dimensionality of each Gaussian, dimensionality of the rU (within client variability matrix) `(#Gaussians, #Inputs, #rU)`.",
  ""
);
PyObject* PyBobLearnEMISVBase_getShape(PyBobLearnEMISVBaseObject* self, void*) {
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
  "WARNING An exception is thrown if no Universal Background Model has been set yet."
  ""
);
PyObject* PyBobLearnEMISVBase_getSupervectorLength(PyBobLearnEMISVBaseObject* self, void*) {
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
PyObject* PyBobLearnEMISVBase_getU(PyBobLearnEMISVBaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getU());
  BOB_CATCH_MEMBER("``u`` could not be read", 0)
}
int PyBobLearnEMISVBase_setU(PyBobLearnEMISVBaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, U.name());
    return -1;
  }
  auto o_ = make_safe(input);
  
  // perform check on the input  
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, U.name());
    return -1;
  }  

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, U.name());
    return -1;
  }  

  if (input->shape[0] != (Py_ssize_t)self->cxx->getU().extent(0) && input->shape[1] != self->cxx->getU().extent(1)) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getU().extent(0), (Py_ssize_t)self->cxx->getU().extent(1), (Py_ssize_t)input->shape[0], (Py_ssize_t)input->shape[1], U.name());
    return -1;
  }  
  
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "u");
  if (!b) return -1;
  self->cxx->setU(*b);
  return 0;
  BOB_CATCH_MEMBER("``u`` matrix could not be set", -1)
}


/***** d *****/
static auto D = bob::extension::VariableDoc(
  "d",
  "array_like <float, 1D>",
  "Returns the diagonal matrix diag(d) (as a 1D vector)",
  ""
);
PyObject* PyBobLearnEMISVBase_getD(PyBobLearnEMISVBaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getD());
  BOB_CATCH_MEMBER("``d`` could not be read", 0)
}
int PyBobLearnEMISVBase_setD(PyBobLearnEMISVBaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 1D array of floats", Py_TYPE(self)->tp_name, D.name());
    return -1;
  }
  auto o_ = make_safe(input);
  
  // perform check on the input  
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, D.name());
    return -1;
  }  

  if (input->ndim != 1){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 1D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, D.name());
    return -1;
  }  

  if (input->shape[0] != (Py_ssize_t)self->cxx->getD().extent(0)) {
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d, elements, not %" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getU().extent(0), (Py_ssize_t)input->shape[0], D.name());
    return -1;
  }  
  
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(input, "d");
  if (!b) return -1;
  self->cxx->setD(*b);
  return 0;
  BOB_CATCH_MEMBER("``d`` matrix could not be set", -1)
}


/***** ubm *****/
static auto ubm = bob::extension::VariableDoc(
  "ubm",
  ":py:class:`bob.learn.em.GMMMachine`",
  "Returns the UBM (Universal Background Model",
  ""
);
PyObject* PyBobLearnEMISVBase_getUBM(PyBobLearnEMISVBaseObject* self, void*){
  BOB_TRY

  boost::shared_ptr<bob::learn::em::GMMMachine> ubm_gmmMachine = self->cxx->getUbm();

  //Allocating the correspondent python object
  PyBobLearnEMGMMMachineObject* retval =
    (PyBobLearnEMGMMMachineObject*)PyBobLearnEMGMMMachine_Type.tp_alloc(&PyBobLearnEMGMMMachine_Type, 0);
  retval->cxx = ubm_gmmMachine;

  return Py_BuildValue("N",retval);
  BOB_CATCH_MEMBER("ubm could not be read", 0)
}
int PyBobLearnEMISVBase_setUBM(PyBobLearnEMISVBaseObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyBobLearnEMGMMMachine_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a :py:class:`bob.learn.em.GMMMachine`", Py_TYPE(self)->tp_name, ubm.name());
    return -1;
  }

  PyBobLearnEMGMMMachineObject* ubm_gmmMachine = 0;
  PyArg_Parse(value, "O!", &PyBobLearnEMGMMMachine_Type,&ubm_gmmMachine);

  self->cxx->setUbm(ubm_gmmMachine->cxx);

  return 0;
  BOB_CATCH_MEMBER("ubm could not be set", -1)  
}



static PyGetSetDef PyBobLearnEMISVBase_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnEMISVBase_getShape,
   0,
   shape.doc(),
   0
  },
  
  {
   supervector_length.name(),
   (getter)PyBobLearnEMISVBase_getSupervectorLength,
   0,
   supervector_length.doc(),
   0
  },
  
  {
   U.name(),
   (getter)PyBobLearnEMISVBase_getU,
   (setter)PyBobLearnEMISVBase_setU,
   U.doc(),
   0
  },
  
  {
   D.name(),
   (getter)PyBobLearnEMISVBase_getD,
   (setter)PyBobLearnEMISVBase_setD,
   D.doc(),
   0
  },

  {
   ubm.name(),
   (getter)PyBobLearnEMISVBase_getUBM,
   (setter)PyBobLearnEMISVBase_setUBM,
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
  "Save the configuration of the ISVBase to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnEMISVBase_Save(PyBobLearnEMISVBaseObject* self,  PyObject* args, PyObject* kwargs) {

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
  "Load the configuration of the ISVBase to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnEMISVBase_Load(PyBobLearnEMISVBaseObject* self, PyObject* args, PyObject* kwargs) {
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
  
  "Compares this ISVBase with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.em.ISVBase`", "A ISVBase object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMISVBase_IsSimilarTo(PyBobLearnEMISVBaseObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMISVBaseObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMISVBase_Type, &other,
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
  "Resets the dimensionality of the subspace U. "
  "U is hence uninitialized.",
  0,
  true
)
.add_prototype("rU")
.add_parameter("rU", "int", "Size of U (Within client variation matrix)");
static PyObject* PyBobLearnEMISVBase_resize(PyBobLearnEMISVBaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  int rU = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &rU)) return 0;

  if (rU <= 0){
    PyErr_Format(PyExc_TypeError, "rU must be greater than zero");
    resize.print_usage();
    return 0;
  }

  self->cxx->resize(rU);

  BOB_CATCH_MEMBER("cannot perform the resize method", 0)

  Py_RETURN_NONE;
}




static PyMethodDef PyBobLearnEMISVBase_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnEMISVBase_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMISVBase_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMISVBase_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnEMISVBase_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the ISV type struct; will be initialized later
PyTypeObject PyBobLearnEMISVBase_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMISVBase(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMISVBase_Type.tp_name      = ISVBase_doc.name();
  PyBobLearnEMISVBase_Type.tp_basicsize = sizeof(PyBobLearnEMISVBaseObject);
  PyBobLearnEMISVBase_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMISVBase_Type.tp_doc       = ISVBase_doc.doc();

  // set the functions
  PyBobLearnEMISVBase_Type.tp_new         = PyType_GenericNew;
  PyBobLearnEMISVBase_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnEMISVBase_init);
  PyBobLearnEMISVBase_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnEMISVBase_delete);
  PyBobLearnEMISVBase_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMISVBase_RichCompare);
  PyBobLearnEMISVBase_Type.tp_methods     = PyBobLearnEMISVBase_methods;
  PyBobLearnEMISVBase_Type.tp_getset      = PyBobLearnEMISVBase_getseters;
  //PyBobLearnEMISVBase_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMISVBase_forward);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMISVBase_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMISVBase_Type);
  return PyModule_AddObject(module, "ISVBase", (PyObject*)&PyBobLearnEMISVBase_Type) >= 0;
}


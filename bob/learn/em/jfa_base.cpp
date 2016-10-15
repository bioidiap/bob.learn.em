/**
 * @date Wed Jan 27 17:03:15 2015 +0200
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
  "A JFABase instance can be seen as a container for :math:`U`, :math:`V` and :math:`D` when performing Joint Factor Analysis (JFA).\n\n"
  "References: [Vogt2008]_ [McCool2013]_",
  ""
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructor. Builds a new JFABase",
    "",
    true
  )
  .add_prototype("ubm,ru,rv","")
  .add_prototype("other","")
  .add_prototype("hdf5","")
  .add_prototype("","")

  .add_parameter("ubm", ":py:class:`bob.learn.em.GMMMachine`", "The Universal Background Model.")
  .add_parameter("ru", "int", "Size of :math:`U` (Within client variation matrix). In the end the U matrix will have (#gaussians * #feature_dimension x ru)")
  .add_parameter("rv", "int", "Size of :math:`V` (Between client variation matrix). In the end the U matrix will have (#gaussians * #feature_dimension x rv)")
  .add_parameter("other", ":py:class:`bob.learn.em.JFABase`", "A JFABase object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnEMJFABase_init_copy(PyBobLearnEMJFABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFABase_doc.kwlist(1);
  PyBobLearnEMJFABaseObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnEMJFABase_Type, &o)){
    JFABase_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::em::JFABase(*o->cxx));
  return 0;
}


static int PyBobLearnEMJFABase_init_hdf5(PyBobLearnEMJFABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFABase_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    JFABase_doc.print_usage();
    return -1;
  }
  auto config_ = make_safe(config);
  self->cxx.reset(new bob::learn::em::JFABase(*(config->f)));

  return 0;
}


static int PyBobLearnEMJFABase_init_ubm(PyBobLearnEMJFABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = JFABase_doc.kwlist(0);
  
  PyBobLearnEMGMMMachineObject* ubm;
  int ru = 1;
  int rv = 1;

  //Here we have to select which keyword argument to read  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ii", kwlist, &PyBobLearnEMGMMMachine_Type, &ubm,
                                                                &ru, &rv)){
    JFABase_doc.print_usage();
    return -1;
  }
  
  if(ru < 0){
    PyErr_Format(PyExc_TypeError, "ru argument must be greater than or equal to one");
    return -1;
  }
  
  if(rv < 0){
    PyErr_Format(PyExc_TypeError, "rv argument must be greater than or equal to one");
    return -1;
  }
  
  self->cxx.reset(new bob::learn::em::JFABase(ubm->cxx, ru, rv));
  return 0;
}


static int PyBobLearnEMJFABase_init(PyBobLearnEMJFABaseObject* self, PyObject* args, PyObject* kwargs) {
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
     if (PyBobLearnEMJFABase_Check(arg))
       return PyBobLearnEMJFABase_init_copy(self, args, kwargs);
      // If the constructor input is a HDF5
     else if (PyBobIoHDF5File_Check(arg))
       return PyBobLearnEMJFABase_init_hdf5(self, args, kwargs);
    }
    case 3:
      return PyBobLearnEMJFABase_init_ubm(self, args, kwargs);
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 3 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
      JFABase_doc.print_usage();
      return -1;
  }
  BOB_CATCH_MEMBER("cannot create JFABase", -1)
  return 0;
}



static void PyBobLearnEMJFABase_delete(PyBobLearnEMJFABaseObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnEMJFABase_RichCompare(PyBobLearnEMJFABaseObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnEMJFABase_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnEMJFABaseObject*>(other);
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

int PyBobLearnEMJFABase_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnEMJFABase_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int, int, int)",
  "A tuple that represents the number of gaussians, dimensionality of each Gaussian, dimensionality of the :math:`rU` (within client variability matrix) and dimensionality of the :math:`rV` (between client variability matrix) ``(#Gaussians, #Inputs, #rU, #rV)``.",
  ""
);
PyObject* PyBobLearnEMJFABase_getShape(PyBobLearnEMJFABaseObject* self, void*) {
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
PyObject* PyBobLearnEMJFABase_getSupervectorLength(PyBobLearnEMJFABaseObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getSupervectorLength());
  BOB_CATCH_MEMBER("supervector_length could not be read", 0)
}


/***** u *****/
static auto U = bob::extension::VariableDoc(
  "u",
  "array_like <float, 2D>",
  "Returns the :math:`U` matrix (within client variability matrix)",
  ""
);
PyObject* PyBobLearnEMJFABase_getU(PyBobLearnEMJFABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getU());
  BOB_CATCH_MEMBER("``u`` could not be read", 0)
}
int PyBobLearnEMJFABase_setU(PyBobLearnEMJFABaseObject* self, PyObject* value, void*){
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

/***** v *****/
static auto V = bob::extension::VariableDoc(
  "v",
  "array_like <float, 2D>",
  "Returns the :math:`V` matrix (between client variability matrix)",
  ""
);
PyObject* PyBobLearnEMJFABase_getV(PyBobLearnEMJFABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getV());
  BOB_CATCH_MEMBER("``v`` could not be read", 0)
}
int PyBobLearnEMJFABase_setV(PyBobLearnEMJFABaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* input;
  if (!PyBlitzArray_Converter(value, &input)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, V.name());
    return -1;
  }
  auto o_ = make_safe(input);
  
  // perform check on the input  
  if (input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `%s`", Py_TYPE(self)->tp_name, V.name());
    return -1;
  }  

  if (input->ndim != 2){
    PyErr_Format(PyExc_TypeError, "`%s' only processes 2D arrays of float64 for `%s`", Py_TYPE(self)->tp_name, V.name());
    return -1;
  }  

  if (input->shape[0] != (Py_ssize_t)self->cxx->getV().extent(0) && input->shape[1] != self->cxx->getV().extent(1)) {
    PyErr_Format(PyExc_TypeError, "`%s' 2D `input` array should have the shape [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] not [%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d] for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getV().extent(0), (Py_ssize_t)self->cxx->getV().extent(1), (Py_ssize_t)input->shape[0], (Py_ssize_t)input->shape[1], V.name());
    return -1;
  }  
  
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(input, "v");
  if (!b) return -1;
  self->cxx->setV(*b);
  return 0;
  BOB_CATCH_MEMBER("``v`` matrix could not be set", -1)
}


/***** d *****/
static auto D = bob::extension::VariableDoc(
  "d",
  "array_like <float, 1D>",
  "Returns the diagonal matrix :math:`diag(d)` (as a 1D vector)",
  ""
);
PyObject* PyBobLearnEMJFABase_getD(PyBobLearnEMJFABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getD());
  BOB_CATCH_MEMBER("``d`` could not be read", 0)
}
int PyBobLearnEMJFABase_setD(PyBobLearnEMJFABaseObject* self, PyObject* value, void*){
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
    PyErr_Format(PyExc_TypeError, "`%s' 1D `input` array should have %" PY_FORMAT_SIZE_T "d elements, not [%" PY_FORMAT_SIZE_T "d for `%s`", Py_TYPE(self)->tp_name, (Py_ssize_t)self->cxx->getD().extent(0), (Py_ssize_t)input->shape[0], D.name());
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
PyObject* PyBobLearnEMJFABase_getUBM(PyBobLearnEMJFABaseObject* self, void*){
  BOB_TRY

  boost::shared_ptr<bob::learn::em::GMMMachine> ubm_gmmMachine = self->cxx->getUbm();

  //Allocating the correspondent python object
  PyBobLearnEMGMMMachineObject* retval =
    (PyBobLearnEMGMMMachineObject*)PyBobLearnEMGMMMachine_Type.tp_alloc(&PyBobLearnEMGMMMachine_Type, 0);
  retval->cxx = ubm_gmmMachine;

  return Py_BuildValue("N",retval);
  BOB_CATCH_MEMBER("ubm could not be read", 0)
}
int PyBobLearnEMJFABase_setUBM(PyBobLearnEMJFABaseObject* self, PyObject* value, void*){
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




static PyGetSetDef PyBobLearnEMJFABase_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnEMJFABase_getShape,
   0,
   shape.doc(),
   0
  },
  
  {
   supervector_length.name(),
   (getter)PyBobLearnEMJFABase_getSupervectorLength,
   0,
   supervector_length.doc(),
   0
  },
  
  {
   U.name(),
   (getter)PyBobLearnEMJFABase_getU,
   (setter)PyBobLearnEMJFABase_setU,
   U.doc(),
   0
  },
  
  {
   V.name(),
   (getter)PyBobLearnEMJFABase_getV,
   (setter)PyBobLearnEMJFABase_setV,
   V.doc(),
   0
  },

  {
   D.name(),
   (getter)PyBobLearnEMJFABase_getD,
   (setter)PyBobLearnEMJFABase_setD,
   D.doc(),
   0
  },

  {
   ubm.name(),
   (getter)PyBobLearnEMJFABase_getUBM,
   (setter)PyBobLearnEMJFABase_setUBM,
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
static PyObject* PyBobLearnEMJFABase_Save(PyBobLearnEMJFABaseObject* self,  PyObject* args, PyObject* kwargs) {

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
static PyObject* PyBobLearnEMJFABase_Load(PyBobLearnEMJFABaseObject* self, PyObject* args, PyObject* kwargs) {
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
.add_parameter("other", ":py:class:`bob.learn.em.JFABase`", "A JFABase object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnEMJFABase_IsSimilarTo(PyBobLearnEMJFABaseObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnEMJFABaseObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnEMJFABase_Type, &other,
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
  "Resets the dimensionality of the subspace U and V. "
  "U and V are hence uninitialized",
  0,
  true
)
.add_prototype("rU,rV")
.add_parameter("rU", "int", "Size of :math:`U` (Within client variation matrix)")
.add_parameter("rV", "int", "Size of :math:`V` (Between client variation matrix)");
static PyObject* PyBobLearnEMJFABase_resize(PyBobLearnEMJFABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  int rU = 0;
  int rV = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &rU, &rV)) return 0;

  if (rU <= 0){
    PyErr_Format(PyExc_TypeError, "rU must be greater than zero");
    resize.print_usage();
    return 0;
  }
  if (rV <= 0){
    PyErr_Format(PyExc_TypeError, "rV must be greater than zero");
    resize.print_usage();
    return 0;
  }
  self->cxx->resize(rU, rV);

  BOB_CATCH_MEMBER("cannot perform the resize method", 0)

  Py_RETURN_NONE;
}




static PyMethodDef PyBobLearnEMJFABase_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnEMJFABase_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnEMJFABase_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnEMJFABase_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnEMJFABase_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the JFA type struct; will be initialized later
PyTypeObject PyBobLearnEMJFABase_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnEMJFABase(PyObject* module)
{
  // initialize the type struct
  PyBobLearnEMJFABase_Type.tp_name      = JFABase_doc.name();
  PyBobLearnEMJFABase_Type.tp_basicsize = sizeof(PyBobLearnEMJFABaseObject);
  PyBobLearnEMJFABase_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnEMJFABase_Type.tp_doc       = JFABase_doc.doc();

  // set the functions
  PyBobLearnEMJFABase_Type.tp_new         = PyType_GenericNew;
  PyBobLearnEMJFABase_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnEMJFABase_init);
  PyBobLearnEMJFABase_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnEMJFABase_delete);
  PyBobLearnEMJFABase_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnEMJFABase_RichCompare);
  PyBobLearnEMJFABase_Type.tp_methods     = PyBobLearnEMJFABase_methods;
  PyBobLearnEMJFABase_Type.tp_getset      = PyBobLearnEMJFABase_getseters;
  //PyBobLearnEMJFABase_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnEMJFABase_forward);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnEMJFABase_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnEMJFABase_Type);
  return PyModule_AddObject(module, "JFABase", (PyObject*)&PyBobLearnEMJFABase_Type) >= 0;
}


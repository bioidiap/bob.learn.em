/**
 * @date Thu Jan 29 15:44:15 2015 +0200
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

static auto PLDABase_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".PLDABase",

  "This class is a container for the :math:`F` (between class variantion matrix), :math:`G` (within class variantion matrix) and :math:`\\Sigma` "
  "matrices and the mean vector :math:`\\mu` of a PLDA model. This also"
  "precomputes useful matrices to make the model scalable."
  "References: [ElShafey2014,PrinceElder2007,LiFu2012]",
  ""
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",

     "Constructor, builds a new PLDABase. :math:`F`, :math:`G` "
     "and :math:`\\Sigma` are initialized to the 'eye' matrix (matrix with 1's "
     "on the diagonal and 0 outside), and :math:`\\mu` is initialized to 0.",

    "",
    true
  )
  .add_prototype("dim_d,dim_f,dim_g,variance_threshold","")
  .add_prototype("other","")
  .add_prototype("hdf5","")

  .add_parameter("dim_D", "int", "Dimensionality of the feature vector.")
  .add_parameter("dim_F", "int", "Size of :math:`F`(between class variantion matrix).")
  .add_parameter("dim_G", "int", "Size of :math:`G`(within class variantion matrix).")
  .add_parameter("variance_threshold", "double", "The smallest possible value of the variance (Ignored if set to 0.)")
  
  .add_parameter("other", ":py:class:`bob.learn.misc.PLDABase`", "A PLDABase object to be copied.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")

);


static int PyBobLearnMiscPLDABase_init_copy(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDABase_doc.kwlist(1);
  PyBobLearnMiscPLDABaseObject* o;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobLearnMiscPLDABase_Type, &o)){
    PLDABase_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::PLDABase(*o->cxx));
  return 0;
}


static int PyBobLearnMiscPLDABase_init_hdf5(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDABase_doc.kwlist(2);

  PyBobIoHDF5FileObject* config = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &config)){
    PLDABase_doc.print_usage();
    return -1;
  }

  self->cxx.reset(new bob::learn::misc::PLDABase(*(config->f)));

  return 0;
}


static int PyBobLearnMiscPLDABase_init_dim(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {

  char** kwlist = PLDABase_doc.kwlist(0);
  
  int dim_D, dim_F, dim_G = 1;
  double variance_threshold = 0.0;

  //Here we have to select which keyword argument to read  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii|d", kwlist, &dim_D, &dim_F, &dim_G, &variance_threshold)){
    PLDABase_doc.print_usage();
    return -1;
  }
  
  if(dim_D <= 0){
    PyErr_Format(PyExc_TypeError, "dim_D argument must be greater than or equal to one");
    return -1;
  }
  
  if(dim_F <= 0){
    PyErr_Format(PyExc_TypeError, "dim_F argument must be greater than or equal to one");
    return -1;
  }

  if(dim_G <= 0){
    PyErr_Format(PyExc_TypeError, "dim_G argument must be greater than or equal to one");
    return -1;
  }

  if(variance_threshold < 0){
    PyErr_Format(PyExc_TypeError, "variance_threshold argument must be greater than or equal to zero");
    return -1;
  }

  
  self->cxx.reset(new bob::learn::misc::PLDABase(dim_D, dim_F, dim_G, variance_threshold));
  return 0;
}

static int PyBobLearnMiscPLDABase_init(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  // get the number of command line arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if(nargs==1){
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
    if (PyBobLearnMiscPLDABase_Check(arg))
      return PyBobLearnMiscPLDABase_init_copy(self, args, kwargs);
    // If the constructor input is a HDF5
    else if (PyBobIoHDF5File_Check(arg))
      return PyBobLearnMiscPLDABase_init_hdf5(self, args, kwargs);
  }
  else if((nargs==3)||(nargs==4))
    return PyBobLearnMiscPLDABase_init_dim(self, args, kwargs);
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1, 3 or 4 arguments, but you provided %d (see help)", Py_TYPE(self)->tp_name, nargs);
    PLDABase_doc.print_usage();
    return -1;
  }
  BOB_CATCH_MEMBER("cannot create PLDABase", 0)
  return 0;
}



static void PyBobLearnMiscPLDABase_delete(PyBobLearnMiscPLDABaseObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnMiscPLDABase_RichCompare(PyBobLearnMiscPLDABaseObject* self, PyObject* other, int op) {
  BOB_TRY

  if (!PyBobLearnMiscPLDABase_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobLearnMiscPLDABaseObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare PLDABase objects", 0)
}

int PyBobLearnMiscPLDABase_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnMiscPLDABase_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

/***** shape *****/
static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int,int, int)",
  "A tuple that represents the dimensionality of the feature vector :math:`dim_d`, the :math:`F` matrix and the :math:`G` matrix.",
  ""
);
PyObject* PyBobLearnMiscPLDABase_getShape(PyBobLearnMiscPLDABaseObject* self, void*) {
  BOB_TRY
  return Py_BuildValue("(i,i,i)", self->cxx->getDimD(), self->cxx->getDimF(), self->cxx->getDimG());
  BOB_CATCH_MEMBER("shape could not be read", 0)
}


/***** F *****/
static auto F = bob::extension::VariableDoc(
  "f",
  "array_like <float, 2D>",
  "Returns the :math:`F` matrix (between class variantion matrix)",
  ""
);
PyObject* PyBobLearnMiscPLDABase_getF(PyBobLearnMiscPLDABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getF());
  BOB_CATCH_MEMBER("`f` could not be read", 0)
}
int PyBobLearnMiscPLDABase_setF(PyBobLearnMiscPLDABaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, F.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(o, "f");
  if (!b) return -1;
  self->cxx->setF(*b);
  return 0;
  BOB_CATCH_MEMBER("`f` vector could not be set", -1)
}

/***** G *****/
static auto G = bob::extension::VariableDoc(
  "g",
  "array_like <float, 2D>",
  "Returns the :math:`G` matrix (between class variantion matrix)",
  ""
);
PyObject* PyBobLearnMiscPLDABase_getG(PyBobLearnMiscPLDABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getG());
  BOB_CATCH_MEMBER("`g` could not be read", 0)
}
int PyBobLearnMiscPLDABase_setG(PyBobLearnMiscPLDABaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, G.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,2>(o, "g");
  if (!b) return -1;
  self->cxx->setG(*b);
  return 0;
  BOB_CATCH_MEMBER("`g` vector could not be set", -1)
}


/***** mu *****/
static auto mu = bob::extension::VariableDoc(
  "mu",
  "array_like <float, 1D>",
  "Gets the :math:`mu` mean vector of the PLDA model",
  ""
);
PyObject* PyBobLearnMiscPLDABase_getMu(PyBobLearnMiscPLDABaseObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getMu());
  BOB_CATCH_MEMBER("`mu` could not be read", 0)
}
int PyBobLearnMiscPLDABase_setMu(PyBobLearnMiscPLDABaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, mu.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "mu");
  if (!b) return -1;
  self->cxx->setMu(*b);
  return 0;
  BOB_CATCH_MEMBER("`mu` vector could not be set", -1)
}


/***** __isigma__ *****/
static auto __isigma__ = bob::extension::VariableDoc(
  "__isigma__",
  "array_like <float, 1D>",
  "Gets the inverse vector/diagonal matrix of :math:`\\Sigma^{-1}`",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getISigma(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getISigma());
  BOB_CATCH_MEMBER("__isigma__ could not be read", 0)
}


/***** __alpha__ *****/
static auto __alpha__ = bob::extension::VariableDoc(
  "__alpha__",
  "array_like <float, 2D>",
  "Gets the \f$\alpha\f$ matrix."
  ":math:`\\alpha = (Id + G^T \\Sigma^{-1} G)^{-1} = \\mathcal{G}`",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getAlpha(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getAlpha());
  BOB_CATCH_MEMBER("__alpha__ could not be read", 0)
}


/***** __beta__ *****/
static auto __beta__ = bob::extension::VariableDoc(
  "__beta__",
  "array_like <float, 2D>",
  "Gets the :math:`\\beta` matrix "
  ":math:`\\beta = (\\Sigma + G G^T)^{-1} = \\mathcal{S} = \\Sigma^{-1} - \\Sigma^{-1} G \\mathcal{G} G^{T} \\Sigma^{-1}`",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getBeta(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getBeta());
  BOB_CATCH_MEMBER("__beta__ could not be read", 0)
}


/***** __ft_beta__ *****/
static auto __ft_beta__ = bob::extension::VariableDoc(
  "__ft_beta__",
  "array_like <float, 2D>",
  "Gets the :math:`F^T \\beta' matrix",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getFtBeta(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getFtBeta());
  BOB_CATCH_MEMBER("__ft_beta__ could not be read", 0)
}


/***** __gt_i_sigma__ *****/
static auto __gt_i_sigma__ = bob::extension::VariableDoc(
  "__gt_i_sigma__",
  "array_like <float, 2D>",
  "Gets the :math:`G^T \\Sigma^{-1}` matrix",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getGtISigma(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getGtISigma());
  BOB_CATCH_MEMBER("__gt_i_sigma__ could not be read", 0)
}


/***** __logdet_alpha__ *****/
static auto __logdet_alpha__ = bob::extension::VariableDoc(
  "__logdet_alpha__",
  "double",
  "Gets :math:`\\log(\\det(\\alpha))`",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getLogDetAlpha(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getLogDetAlpha());
  BOB_CATCH_MEMBER("__logdet_alpha__ could not be read", 0)
}

/***** __logdet_sigma__ *****/
static auto __logdet_sigma__ = bob::extension::VariableDoc(
  "__logdet_sigma__",
  "double",
  "Gets :math:`\\log(\\det(\\Sigma))`",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getLogDetSigma(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getLogDetSigma());
  BOB_CATCH_MEMBER("__logdet_sigma__ could not be read", 0)
}


/***** variance_threshold *****/
static auto variance_threshold = bob::extension::VariableDoc(
  "variance_threshold",
  "double",
  "",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getVarianceThreshold(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return Py_BuildValue("d",self->cxx->getVarianceThreshold());
  BOB_CATCH_MEMBER("variance_threshold could not be read", 0)
}
int PyBobLearnMiscPLDABase_setVarianceThreshold(PyBobLearnMiscPLDABaseObject* self, PyObject* value, void*){
  BOB_TRY

  if (!PyNumber_Check(value)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects an float", Py_TYPE(self)->tp_name, variance_threshold.name());
    return -1;
  }

  self->cxx->setVarianceThreshold(PyFloat_AS_DOUBLE(value));
  BOB_CATCH_MEMBER("variance_threshold could not be set", -1)
  return 0;
}




/***** sigma *****/
static auto sigma = bob::extension::VariableDoc(
  "sigma",
  "array_like <float, 1D>",
  "Gets the :math:`\\sigma` (diagonal) covariance matrix of the PLDA model",
  ""
);
static PyObject* PyBobLearnMiscPLDABase_getSigma(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getSigma());
  BOB_CATCH_MEMBER("sigma could not be read", 0)
}
int PyBobLearnMiscPLDABase_setSigma(PyBobLearnMiscPLDABaseObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* o;
  if (!PyBlitzArray_Converter(value, &o)){
    PyErr_Format(PyExc_RuntimeError, "%s %s expects a 2D array of floats", Py_TYPE(self)->tp_name, sigma.name());
    return -1;
  }
  auto o_ = make_safe(o);
  auto b = PyBlitzArrayCxx_AsBlitz<double,1>(o, "sigma");
  if (!b) return -1;
  self->cxx->setSigma(*b);
  return 0;
  BOB_CATCH_MEMBER("`sigma` vector could not be set", -1)
}


static PyGetSetDef PyBobLearnMiscPLDABase_getseters[] = { 
  {
   shape.name(),
   (getter)PyBobLearnMiscPLDABase_getShape,
   0,
   shape.doc(),
   0
  },  
  {
   F.name(),
   (getter)PyBobLearnMiscPLDABase_getF,
   (setter)PyBobLearnMiscPLDABase_setF,
   F.doc(),
   0
  },
  {
   G.name(),
   (getter)PyBobLearnMiscPLDABase_getG,
   (setter)PyBobLearnMiscPLDABase_setG,
   G.doc(),
   0
  },
  {
   mu.name(),
   (getter)PyBobLearnMiscPLDABase_getMu,
   (setter)PyBobLearnMiscPLDABase_setMu,
   mu.doc(),
   0
  },
  {
   __isigma__.name(),
   (getter)PyBobLearnMiscPLDABase_getISigma,
   0,
   __isigma__.doc(),
   0
  },
  {
   __alpha__.name(),
   (getter)PyBobLearnMiscPLDABase_getAlpha,
   0,
   __alpha__.doc(),
   0
  },
  {
   __beta__.name(),
   (getter)PyBobLearnMiscPLDABase_getBeta,
   0,
   __beta__.doc(),
   0
  },
  {
  __ft_beta__.name(),
   (getter)PyBobLearnMiscPLDABase_getFtBeta,
   0,
   __ft_beta__.doc(),
   0
  },
  {
  __gt_i_sigma__.name(),
   (getter)PyBobLearnMiscPLDABase_getGtISigma,
   0,
   __gt_i_sigma__.doc(),
   0
  },
  {
  __logdet_alpha__.name(),
   (getter)PyBobLearnMiscPLDABase_getLogDetAlpha,
   0,
   __logdet_alpha__.doc(),
   0
  },
  {
  __logdet_sigma__.name(),
   (getter)PyBobLearnMiscPLDABase_getLogDetSigma,
   0,
   __logdet_sigma__.doc(),
   0
  },
  {
   sigma.name(),
   (getter)PyBobLearnMiscPLDABase_getSigma,
   (setter)PyBobLearnMiscPLDABase_setSigma,
   sigma.doc(),
   0
  },
  {
   variance_threshold.name(),
   (getter)PyBobLearnMiscPLDABase_getVarianceThreshold,
   (setter)PyBobLearnMiscPLDABase_setVarianceThreshold,
   variance_threshold.doc(),
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
  "Save the configuration of the PLDABase to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing");
static PyObject* PyBobLearnMiscPLDABase_Save(PyBobLearnMiscPLDABaseObject* self,  PyObject* args, PyObject* kwargs) {

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
  "Load the configuration of the PLDABase to a given HDF5 file"
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading");
static PyObject* PyBobLearnMiscPLDABase_Load(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
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
  
  "Compares this PLDABase with the ``other`` one to be approximately the same.",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the "
  "relative and absolute precision for the ``weights``, ``biases`` "
  "and any other values internal to this machine."
)
.add_prototype("other, [r_epsilon], [a_epsilon]","output")
.add_parameter("other", ":py:class:`bob.learn.misc.PLDABase`", "A PLDABase object to be compared.")
.add_parameter("r_epsilon", "float", "Relative precision.")
.add_parameter("a_epsilon", "float", "Absolute precision.")
.add_return("output","bool","True if it is similar, otherwise false.");
static PyObject* PyBobLearnMiscPLDABase_IsSimilarTo(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist(0);

  //PyObject* other = 0;
  PyBobLearnMiscPLDABaseObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnMiscPLDABase_Type, &other,
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
  "Resizes the dimensionality of the PLDA model. Paramaters :math:`\\mu`, :math:`\\F`, :math:`\\G` and :math:`\\Sigma` are reinitialized.",
  0,
  true
)
.add_prototype("dim_D,dim_F,dim_G")
.add_parameter("dim_D", "int", "Dimensionality of the feature vector.")
.add_parameter("dim_F", "int", "Size of :math:`F`(between class variantion matrix).")
.add_parameter("dim_G", "int", "Size of :math:`F`(within class variantion matrix).");
static PyObject* PyBobLearnMiscPLDABase_resize(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist(0);

  int dim_D, dim_F, dim_G = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii", kwlist, &dim_D, &dim_F, &dim_G)) Py_RETURN_NONE;

  if(dim_D <= 0){
    PyErr_Format(PyExc_TypeError, "dim_D argument must be greater than or equal to one");
    Py_RETURN_NONE;
  }
  
  if(dim_F <= 0){
    PyErr_Format(PyExc_TypeError, "dim_F argument must be greater than or equal to one");
    Py_RETURN_NONE;
  }

  if(dim_G <= 0){
    PyErr_Format(PyExc_TypeError, "dim_G argument must be greater than or equal to one");
    Py_RETURN_NONE;
  }

  self->cxx->resize(dim_D, dim_F, dim_G);

  BOB_CATCH_MEMBER("cannot perform the resize method", 0)

  Py_RETURN_NONE;
}


/***** get_gamma *****/
static auto get_gamma = bob::extension::FunctionDoc(
  "get_gamma",
  "Gets the :math:`\\gamma_a` matrix for a given :math:`a` (number of samples). "
  ":math:`gamma_{a} = (Id + a F^T \beta F)^{-1} = \\mathcal{F}_{a}`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","array_like <float, 2D>","Get the :math:`\\gamma` matrix");
static PyObject* PyBobLearnMiscPLDABase_getGamma(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_gamma.kwlist(0);

  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getGamma(i));
  BOB_CATCH_MEMBER("`get_gamma` could not be read", 0)
}


/***** has_gamma *****/
static auto has_gamma = bob::extension::FunctionDoc(
  "has_gamma",
  "Tells if the :math:`gamma_a` matrix for a given a (number of samples) exists. "
  ":math:`gamma_a=(Id + a F^T \\beta F)^{-1}`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","bool","");
static PyObject* PyBobLearnMiscPLDABase_hasGamma(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = has_gamma.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  if(self->cxx->hasGamma(i))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
 BOB_CATCH_MEMBER("`has_gamma` could not be read", 0)    
}


/***** compute_gamma *****/
static auto compute_gamma = bob::extension::FunctionDoc(
  "compute_gamma",
  "Tells if the :math:`gamma_a` matrix for a given a (number of samples) exists."
  ":math:`gamma_a = (Id + a F^T \\beta F)^{-1}`",
  0,
  true
)
.add_prototype("a,res","")
.add_parameter("a", "int", "Index")
.add_parameter("res", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnMiscPLDABase_computeGamma(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = compute_gamma.kwlist(0);
  int i = 0;
  PyBlitzArrayObject* res = 0;  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO&", kwlist, &i, &PyBlitzArray_Converter, &res)) Py_RETURN_NONE;

  auto res_ = make_safe(res);  

  self->cxx->computeGamma(i,*PyBlitzArrayCxx_AsBlitz<double,2>(res));
  Py_RETURN_NONE;
  BOB_CATCH_MEMBER("`compute_gamma` could not be read", 0)    
}

/***** get_add_gamma *****/
static auto get_add_gamma = bob::extension::FunctionDoc(
  "get_add_gamma",
   "Gets the :math:`gamma_a` matrix for a given :math:`f_a` (number of samples)."
   ":math:`gamma_a = (Id + a F^T \\beta F)^{-1} = \\mathcal{F}_{a}`."
   "Tries to find it from the base machine and then from this machine.",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","array_like <float, 2D>","");
static PyObject* PyBobLearnMiscPLDABase_getAddGamma(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_add_gamma.kwlist(0);

  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getAddGamma(i));
  BOB_CATCH_MEMBER("`get_add_gamma` could not be read", 0)
}


/***** has_log_like_const_term *****/
static auto has_log_like_const_term = bob::extension::FunctionDoc(
  "has_log_like_const_term",
   "Tells if the log likelihood constant term for a given :math:`a` (number of samples) exists in this machine (does not check the base machine). "
   ":math:`l_{a}=\\frac{a}{2} ( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| +log|\\gamma_a|)`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","bool","");
static PyObject* PyBobLearnMiscPLDABase_hasLogLikeConstTerm(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = has_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  if(self->cxx->hasLogLikeConstTerm(i))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
 BOB_CATCH_MEMBER("`has_log_like_const_term` could not be read", 0)    
}


/***** compute_log_like_const_term" *****/
static auto compute_log_like_const_term = bob::extension::FunctionDoc(
  "compute_log_like_const_term",
  "Computes the log likelihood constant term for a given :math:`a` (number of samples), given the provided :math:`gamma_a` matrix. "
  ":math:`l_{a} = \\frac{a}{2} ( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| +log|\\gamma_a|)`",

  0,
  true
)
.add_prototype("a,res","")
.add_parameter("a", "int", "Index")
.add_parameter("res", "array_like <float, 2D>", "Input data");
static PyObject* PyBobLearnMiscPLDABase_computeLogLikeConstTerm(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = compute_log_like_const_term.kwlist(0);
  int i = 0;
  PyBlitzArrayObject* res = 0;  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO&", kwlist, &i, &PyBlitzArray_Converter, &res)) Py_RETURN_NONE;

  auto res_ = make_safe(res);  

  self->cxx->computeLogLikeConstTerm(i,*PyBlitzArrayCxx_AsBlitz<double,2>(res));
  Py_RETURN_NONE;
  BOB_CATCH_MEMBER("`compute_gamma` could not be read", 0)    
}


/***** get_add_log_like_const_term *****/
static auto get_add_log_like_const_term = bob::extension::FunctionDoc(
  "get_add_log_like_const_term",

   "Gets the log likelihood constant term for a given :math:`a` (number of samples). "
   ":math:`l_{a} = \\frac{a}{2} ( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| +log|\\gamma_a|)`",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","double","");
static PyObject* PyBobLearnMiscPLDABase_getAddLogLikeConstTerm(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_add_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  return Py_BuildValue("d",self->cxx->getAddLogLikeConstTerm(i));

  BOB_CATCH_MEMBER("`get_add_log_like_const_term` could not be read", 0)    
}


/***** get_log_like_const_term *****/
static auto get_log_like_const_term = bob::extension::FunctionDoc(
  "get_log_like_const_term",
   "Gets the log likelihood constant term for a given :math:`a` (number of samples). "
    ":math:`l_{a} = \\frac{a}{2} ( -D log(2\\pi) -log|\\Sigma| +log|\\alpha| +log|\\gamma_a|)",
  0,
  true
)
.add_prototype("a","output")
.add_parameter("a", "int", "Index")
.add_return("output","double","");
static PyObject* PyBobLearnMiscPLDABase_getLogLikeConstTerm(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = get_log_like_const_term.kwlist(0);
  int i = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &i)) Py_RETURN_NONE;

  return Py_BuildValue("d",self->cxx->getLogLikeConstTerm(i));

  BOB_CATCH_MEMBER("`get_log_like_const_term` could not be read", 0)    
}

/***** clear_maps *****/
static auto clear_maps = bob::extension::FunctionDoc(
  "clear_maps",
  "Clears the maps (:math:`gamma_a` and loglike_constterm_a).",
  0,
  true
);
static PyObject* PyBobLearnMiscPLDABase_clearMaps(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  self->cxx->clearMaps();
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("`clear_maps` could not be read", 0)    
}


/***** compute_log_likelihood_point_estimate *****/
static auto compute_log_likelihood_point_estimate = bob::extension::FunctionDoc(
  "compute_log_likelihood_point_estimate",
   "Gets the log-likelihood of an observation, given the current model and the latent variables (point estimate)."
   "This will basically compute :math:`p(x_{ij} | h_{i}, w_{ij}, \\Theta)`, given by "
   ":math:`\\mathcal{N}(x_{ij}|[\\mu + F h_{i} + G w_{ij} + \\epsilon_{ij}, \\Sigma])`, which is in logarithm, "
   ":math:`\\frac{D}{2} log(2\\pi) -\\frac{1}{2} log(det(\\Sigma)) -\\frac{1}{2} {(x_{ij}-(\\mu+F h_{i}+G w_{ij}))^{T}\\Sigma^{-1}(x_{ij}-(\\mu+F h_{i}+G w_{ij}))}`",
  0,
  true
)
.add_prototype("xij,hi,wij","output")
.add_parameter("xij", "array_like <float, 1D>", "")
.add_parameter("hi", "array_like <float, 1D>", "")
.add_parameter("wij", "array_like <float, 1D>", "")
.add_return("output", "double", "");
static PyObject* PyBobLearnMiscPLDABase_computeLogLikelihoodPointEstimate(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  char** kwlist = compute_log_likelihood_point_estimate.kwlist(0);
  PyBlitzArrayObject* xij, *hi, *wij;  
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&", kwlist, &PyBlitzArray_Converter, &xij,
                                                               &PyBlitzArray_Converter, &hi,
                                                               &PyBlitzArray_Converter, &wij)) return 0;

  auto xij_ = make_safe(xij);
  auto hi_ = make_safe(hi);
  auto wij_ = make_safe(wij);  

  return Py_BuildValue("d", self->cxx->computeLogLikelihoodPointEstimate(*PyBlitzArrayCxx_AsBlitz<double,1>(xij), *PyBlitzArrayCxx_AsBlitz<double,1>(hi), *PyBlitzArrayCxx_AsBlitz<double,1>(wij)));
  
  BOB_CATCH_MEMBER("`compute_log_likelihood_point_estimate` could not be read", 0)    
}

/***** __precompute__ *****/
static auto __precompute__ = bob::extension::FunctionDoc(
  "__precompute__",
  "Precomputes useful values for the log likelihood "
  ":math:`\\log(\\det(\\alpha))` and :math:`\\log(\\det(\\Sigma))`.",
  0,
  true
);
static PyObject* PyBobLearnMiscPLDABase_precompute(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  self->cxx->precompute();
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("`precompute` could not be read", 0)    
}


/***** __precompute_log_like__ *****/
static auto __precompute_log_like__ = bob::extension::FunctionDoc(
  "__precompute_log_like__",

  "Precomputes useful values for the log likelihood "
  ":math:`\\log(\\det(\\alpha))` and :math:`\\log(\\det(\\Sigma))`.",

  0,
  true
);
static PyObject* PyBobLearnMiscPLDABase_precomputeLogLike(PyBobLearnMiscPLDABaseObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  
  self->cxx->precomputeLogLike();
  Py_RETURN_NONE;

  BOB_CATCH_MEMBER("`__precompute_log_like__` could not be read", 0)    
}


static PyMethodDef PyBobLearnMiscPLDABase_methods[] = {
  {
    save.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
  },
  {
    get_gamma.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_getGamma,
    METH_VARARGS|METH_KEYWORDS,
    get_gamma.doc()
  },
  {
    has_gamma.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_hasGamma,
    METH_VARARGS|METH_KEYWORDS,
    has_gamma.doc()
  },
  {
    compute_gamma.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_computeGamma,
    METH_VARARGS|METH_KEYWORDS,
    compute_gamma.doc()
  },
  {
    get_add_gamma.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_getAddGamma,
    METH_VARARGS|METH_KEYWORDS,
    get_add_gamma.doc()
  },
  {
    has_log_like_const_term.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_hasLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    has_log_like_const_term.doc()
  },  
  {
    compute_log_like_const_term.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_computeLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    compute_log_like_const_term.doc()
  },  
  {
    get_add_log_like_const_term.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_getAddLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    get_add_log_like_const_term.doc()
  },  
  {
    get_log_like_const_term.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_getLogLikeConstTerm,
    METH_VARARGS|METH_KEYWORDS,
    get_log_like_const_term.doc()
  },  
  {
    clear_maps.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_clearMaps,
    METH_NOARGS,
    clear_maps.doc()
  },
  {
    compute_log_likelihood_point_estimate.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_computeLogLikelihoodPointEstimate,
    METH_VARARGS|METH_KEYWORDS,
    compute_log_likelihood_point_estimate.doc()
  },
  {
    __precompute__.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_precompute,
    METH_NOARGS,
    __precompute__.doc()
  },   
  {
    __precompute_log_like__.name(),
    (PyCFunction)PyBobLearnMiscPLDABase_precomputeLogLike,
    METH_NOARGS,
    __precompute_log_like__.doc()
  },     
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the JFA type struct; will be initialized later
PyTypeObject PyBobLearnMiscPLDABase_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscPLDABase(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscPLDABase_Type.tp_name      = PLDABase_doc.name();
  PyBobLearnMiscPLDABase_Type.tp_basicsize = sizeof(PyBobLearnMiscPLDABaseObject);
  PyBobLearnMiscPLDABase_Type.tp_flags     = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscPLDABase_Type.tp_doc       = PLDABase_doc.doc();

  // set the functions
  PyBobLearnMiscPLDABase_Type.tp_new         = PyType_GenericNew;
  PyBobLearnMiscPLDABase_Type.tp_init        = reinterpret_cast<initproc>(PyBobLearnMiscPLDABase_init);
  PyBobLearnMiscPLDABase_Type.tp_dealloc     = reinterpret_cast<destructor>(PyBobLearnMiscPLDABase_delete);
  PyBobLearnMiscPLDABase_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnMiscPLDABase_RichCompare);
  PyBobLearnMiscPLDABase_Type.tp_methods     = PyBobLearnMiscPLDABase_methods;
  PyBobLearnMiscPLDABase_Type.tp_getset      = PyBobLearnMiscPLDABase_getseters;
  //PyBobLearnMiscPLDABase_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnMiscPLDABase_forward);


  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscPLDABase_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscPLDABase_Type);
  return PyModule_AddObject(module, "PLDABase", (PyObject*)&PyBobLearnMiscPLDABase_Type) >= 0;
}


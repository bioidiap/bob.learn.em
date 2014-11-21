/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Fri 21 Nov 10:38:48 2013
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto Gaussian_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".Gaussian",
  "This class implements a multivariate diagonal Gaussian distribution"
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new multivariate gaussian object",
    "",
    true
  )
  .add_prototype("[n_inputs]")
  //.add_prototype("tan_triggs", "")
  //.add_parameter("gamma", "float", "[default: ``0.2``] The value of gamma for the gamma correction")
  //.add_parameter("sigma0", "float", "[default: ``1.``] The standard deviation of the inner Gaussian")
  //.add_parameter("sigma1", "float", "[default: ``2.``] The standard deviation of the outer Gaussian")
  //.add_parameter("radius", "int", "[default: ``2``] The radius of the Difference of Gaussians filter along both axes (size of the kernel=2*radius+1)")
  //.add_parameter("threshold", "float", "[default: ``10.``] The threshold used for the contrast equalization")
  //.add_parameter("alpha", "float", "[default: ``0.1``] The alpha value used for the contrast equalization")
  //.add_parameter("border", ":py:class:`bob.sp.BorderType`", "[default: ``bob.sp.BorderType.Mirror``] The extrapolation method used by the convolution at the border")
  //.add_parameter("tan_triggs", ":py:class:`bob.ip.base.TanTriggs`", "The TanTriggs object to use for copy-construction")
);


static int PyBobLearnMiscGaussian_init(PyBobLearnMiscGaussianObject* self, PyObject* args, PyObject* kwargs) {
  TRY

  char* kwlist1[] = {c("n_inputs")};
  //char* kwlist2[] = {c("tan_triggs"), NULL};

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);


  /*
  PyObject* k = Py_BuildValue("s", kwlist2[0]);
  auto k_ = make_safe(k);
  if (nargs == 1 && ((args && PyTuple_Size(args) == 1 && PyBobIpBaseTanTriggs_Check(PyTuple_GET_ITEM(args,0))) || (kwargs && PyDict_Contains(kwargs, k)))){
    // copy construct
    PyBobIpBaseTanTriggsObject* tt;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpBaseTanTriggs_Type, &tt)) return -1;

    self->cxx.reset(new bob::ip::base::TanTriggs(*tt->cxx));
    return 0;
  }*/

  size_t n_inputs=0;
  self->cxx.reset(new bob::learn::misc::Gaussian(n_inputs));
  return 0;

  CATCH("cannot create Gaussian", -1)
}


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the Gaussian type struct; will be initialized later
PyTypeObject PyBobLearnMiscGaussian_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnMiscGaussian(PyObject* module)
{
  // initialize the type struct
  PyBobLearnMiscGaussian_Type.tp_name = Gaussian_doc.name();
  PyBobLearnMiscGaussian_Type.tp_basicsize = sizeof(PyBobLearnMiscGaussianObject);
  PyBobLearnMiscGaussian_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnMiscGaussian_Type.tp_doc = Gaussian_doc.doc();

  // set the functions
  PyBobLearnMiscGaussian_Type.tp_new = PyType_GenericNew;
  PyBobLearnMiscGaussian_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnMiscGaussian_init);
  //PyBobLearnMiscGaussian_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpBaseTanTriggs_delete);
  //PyBobLearnMiscGaussian_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpBaseTanTriggs_RichCompare);
  //PyBobLearnMiscGaussian_Type.tp_methods = PyBobIpBaseTanTriggs_methods;
  //PyBobLearnMiscGaussian_Type.tp_getset = PyBobIpBaseTanTriggs_getseters;
  //PyBobLearnMiscGaussian_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpBaseTanTriggs_process);

  // check that everything is fine
  if (PyType_Ready(&PyBobLearnMiscGaussian_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnMiscGaussian_Type);
  return PyModule_AddObject(module, "Gaussian", (PyObject*)&PyBobLearnMiscGaussian_Type) >= 0;
}


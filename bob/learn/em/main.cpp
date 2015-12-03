/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Fri Nov 21 12:39:21 CET 2014
 *
 * @brief Bindings to bob::learn::em routines
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include "main.h"

static PyMethodDef module_methods[] = {
  {
    zt_norm.name(),
    (PyCFunction)PyBobLearnEM_ztNorm,
    METH_VARARGS|METH_KEYWORDS,
    zt_norm.doc()
  },
  {
    t_norm.name(),
    (PyCFunction)PyBobLearnEM_tNorm,
    METH_VARARGS|METH_KEYWORDS,
    t_norm.doc()
  },
  {
    z_norm.name(),
    (PyCFunction)PyBobLearnEM_zNorm,
    METH_VARARGS|METH_KEYWORDS,
    z_norm.doc()
  },
  {
    linear_scoring1.name(),
    (PyCFunction)PyBobLearnEM_linear_scoring,
    METH_VARARGS|METH_KEYWORDS,
    linear_scoring1.doc()
  },

  {0}//Sentinel
};


PyDoc_STRVAR(module_docstr, "Bob EM based Machine Learning Routines");

int PyBobLearnEM_APIVersion = BOB_LEARN_EM_API_VERSION;


#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  if (!init_BobLearnEMGaussian(module)) return 0;
  if (!init_BobLearnEMGMMStats(module)) return 0;
  if (!init_BobLearnEMGMMMachine(module)) return 0;
  if (!init_BobLearnEMKMeansMachine(module)) return 0;
  if (!init_BobLearnEMKMeansTrainer(module)) return 0;
  if (!init_BobLearnEMMLGMMTrainer(module)) return 0;
  if (!init_BobLearnEMMAPGMMTrainer(module)) return 0;

  if (!init_BobLearnEMJFABase(module)) return 0;
  if (!init_BobLearnEMJFAMachine(module)) return 0;
  if (!init_BobLearnEMJFATrainer(module)) return 0;

  if (!init_BobLearnEMISVBase(module)) return 0;
  if (!init_BobLearnEMISVMachine(module)) return 0;
  if (!init_BobLearnEMISVTrainer(module)) return 0;

  if (!init_BobLearnEMIVectorMachine(module)) return 0;
  if (!init_BobLearnEMIVectorTrainer(module)) return 0;

  if (!init_BobLearnEMPLDABase(module)) return 0;
  if (!init_BobLearnEMPLDAMachine(module)) return 0;
  if (!init_BobLearnEMPLDATrainer(module)) return 0;

  if (!init_BobLearnEMEMPCATrainer(module)) return 0;


  static void* PyBobLearnEM_API[PyBobLearnEM_API_pointers];

  /* exhaustive list of C APIs */

  /**************
   * Versioning *
   **************/

  PyBobLearnEM_API[PyBobLearnEM_APIVersion_NUM] = (void *)&PyBobLearnEM_APIVersion;


#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobLearnEM_API,
      BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobLearnEM_API, 0);

#endif

  if (!c_api_object) return 0;

  if (PyModule_AddObject(module, "_C_API", c_api_object) < 0) return 0;

  /* imports bob.learn.em's C-API dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_core_random() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;
  if (import_bob_sp() < 0) return 0;
  if (import_bob_learn_activation() < 0) return 0;
  if (import_bob_learn_linear() < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}

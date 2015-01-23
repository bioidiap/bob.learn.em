/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Fri Nov 21 12:39:21 CET 2014
 *
 * @brief Bindings to bob::learn::misc routines
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include "main.h"

static PyMethodDef module_methods[] = {};


PyDoc_STRVAR(module_docstr, "Bob EM based Machine Learning Routines");

int PyBobLearnMisc_APIVersion = BOB_LEARN_MISC_API_VERSION;


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
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!module) return 0;
  auto module_ = make_safe(module); ///< protects against early returns

  if (PyModule_AddStringConstant(module, "__version__", BOB_EXT_MODULE_VERSION) < 0) return 0;
  if (!init_BobLearnMiscGaussian(module)) return 0;
  if (!init_BobLearnMiscGMMStats(module)) return 0;
  if (!init_BobLearnMiscGMMMachine(module)) return 0;
  if (!init_BobLearnMiscKMeansMachine(module)) return 0;
  if (!init_BobLearnMiscKMeansTrainer(module)) return 0;
  if (!init_BobLearnMiscGMMBaseTrainer(module)) return 0;
  if (!init_BobLearnMiscMLGMMTrainer(module)) return 0;  
  if (!init_BobLearnMiscMAPGMMTrainer(module)) return 0;


  static void* PyBobLearnMisc_API[PyBobLearnMisc_API_pointers];

  /* exhaustive list of C APIs */

  /**************
   * Versioning *
   **************/

  PyBobLearnMisc_API[PyBobLearnMisc_APIVersion_NUM] = (void *)&PyBobLearnMisc_APIVersion;


#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobLearnMisc_API,
      BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobLearnMisc_API, 0);

#endif

  if (!c_api_object) return 0;

  if (PyModule_AddObject(module, "_C_API", c_api_object) < 0) return 0;


  /* imports bob.learn.misc's C-API dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_random() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;

  Py_INCREF(module);
  return module;

}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}

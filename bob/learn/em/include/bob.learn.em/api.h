/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Fri 21 Nov 10:38:48 2013
 *
 * @brief Python API for bob::learn::em
 */

#ifndef BOB_LEARN_EM_API_H
#define BOB_LEARN_EM_API_H

/* Define Module Name and Prefix for other Modules
   Note: We cannot use BOB_EXT_* macros here, unfortunately */
#define BOB_LEARN_EM_PREFIX    "bob.learn.em"
#define BOB_LEARN_EM_FULL_NAME "bob.learn.em._library"

#include <Python.h>

#include <bob.learn.em/config.h>
#include <boost/shared_ptr.hpp>

/*******************
 * C API functions *
 *******************/

/* Enum defining entries in the function table */
enum _PyBobLearnEM_ENUM{
  PyBobLearnEM_APIVersion_NUM = 0,
  // bindings
  ////PyBobIpBaseLBP_Type_NUM,
  ////PyBobIpBaseLBP_Check_NUM,
  ////PyBobIpBaseLBP_Converter_NUM,
  // Total number of C API pointers
  PyBobLearnEM_API_pointers
};


#ifdef BOB_LEARN_EM_MODULE

  /* This section is used when compiling `bob.io.base' itself */

  /**************
   * Versioning *
   **************/

  extern int PyBobLearnEM_APIVersion;

#else // BOB_LEARN_EM_MODULE

  /* This section is used in modules that use `bob.io.base's' C-API */

#if defined(NO_IMPORT_ARRAY)
  extern void **PyBobLearnEM_API;
#elif defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyBobLearnEM_API;
#else
  static void **PyBobLearnEM_API=NULL;
#endif

  /**************
   * Versioning *
   **************/

#define PyBobLearnEM_APIVersion (*(int *)PyBobLearnEM_API[PyBobLearnEM_APIVersion_NUM])

#if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success.
   */
  static int import_bob_learn_em(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOB_LEARN_EM_FULL_NAME);

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyBobLearnEM_API = (void **)PyCapsule_GetPointer(c_api_object, PyCapsule_GetName(c_api_object));
    }
#else
    if (PyCObject_Check(c_api_object)) {
      PyBobLearnEM_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyBobLearnEM_API) {
      PyErr_SetString(PyExc_ImportError, "cannot find C/C++ API "
#if PY_VERSION_HEX >= 0x02070000
          "capsule"
#else
          "cobject"
#endif
          " at `" BOB_LEARN_EM_FULL_NAME "._C_API'");
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyBobLearnEM_API[PyBobLearnEM_APIVersion_NUM];

    if (BOB_LEARN_EM_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, BOB_LEARN_EM_FULL_NAME " import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOB_LEARN_EM_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

#endif //!defined(NO_IMPORT_ARRAY)

#endif /* BOB_LEARN_EM_MODULE */

#endif /* BOB_LEARN_EM_API_H */

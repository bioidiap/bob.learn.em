/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Mon Apr 14 20:43:48 CEST 2014
 *
 * @brief Binds configuration information available from bob
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>

#include <bob.core/config.h>
#include <bob.io.base/config.h>
#include <bob.sp/config.h>
#include <bob.math/config.h>
#include <bob.learn.activation/config.h>
#include <bob.learn.linear/config.h>
// TODO: add other dependencies

#include <string>
#include <cstdlib>
#include <blitz/blitz.h>
#include <boost/preprocessor/stringize.hpp>
#include <boost/version.hpp>
#include <boost/format.hpp>


static int dict_set(PyObject* d, const char* key, const char* value) {
  PyObject* v = Py_BuildValue("s", value);
  if (!v) return 0;
  int retval = PyDict_SetItemString(d, key, v);
  Py_DECREF(v);
  if (retval == 0) return 1; //all good
  return 0; //a problem occurred
}

static int dict_steal(PyObject* d, const char* key, PyObject* value) {
  if (!value) return 0;
  int retval = PyDict_SetItemString(d, key, value);
  Py_DECREF(value);
  if (retval == 0) return 1; //all good
  return 0; //a problem occurred
}

/**
 * Describes the version of Boost libraries installed
 */
static PyObject* boost_version() {
  boost::format f("%d.%d.%d");
  f % (BOOST_VERSION / 100000);
  f % (BOOST_VERSION / 100 % 1000);
  f % (BOOST_VERSION % 100);
  return Py_BuildValue("s", f.str().c_str());
}

/**
 * Describes the compiler version
 */
static PyObject* compiler_version() {
# if defined(__GNUC__) && !defined(__llvm__)
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(__GNUC__);
  f % BOOST_PP_STRINGIZE(__GNUC_MINOR__);
  f % BOOST_PP_STRINGIZE(__GNUC_PATCHLEVEL__);
  return Py_BuildValue("ss", "gcc", f.str().c_str());
# elif defined(__llvm__) && !defined(__clang__)
  return Py_BuildValue("ss", "llvm-gcc", __VERSION__);
# elif defined(__clang__)
  return Py_BuildValue("ss", "clang", __clang_version__);
# else
  return Py_BuildValue("s", "unsupported");
# endif
}

/**
 * Python version with which we compiled the extensions
 */
static PyObject* python_version() {
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(PY_MAJOR_VERSION);
  f % BOOST_PP_STRINGIZE(PY_MINOR_VERSION);
  f % BOOST_PP_STRINGIZE(PY_MICRO_VERSION);
  return Py_BuildValue("s", f.str().c_str());
}

/**
 * Numpy version
 */
static PyObject* numpy_version() {
  return Py_BuildValue("{ssss}", "abi", BOOST_PP_STRINGIZE(NPY_VERSION),
      "api", BOOST_PP_STRINGIZE(NPY_API_VERSION));
}

/**
 * bob.blitz c/c++ api version
 */
static PyObject* bob_blitz_version() {
  return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_BLITZ_API_VERSION));
}

/**
 * bob.core c/c++ api version
 */
static PyObject* bob_core_version() {
  return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_CORE_API_VERSION));
}

/**
 * bob.io.base c/c++ api version
 */
static PyObject* bob_io_base_version() {
  return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_IO_BASE_API_VERSION));
}

/**
 * bob.sp c/c++ api version
 */
static PyObject* bob_sp_version() {
  return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_SP_API_VERSION));
}

/**
 * bob.math c/c++ api version
 */
static PyObject* bob_math_version() {
  return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_MATH_API_VERSION));
}

/**
 * bob.learn.activation c/c++ api version
 */
static PyObject* bob_learn_activation_version() {
  return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_LEARN_ACTIVATION_API_VERSION));
}

/**
 * bob.learn.linear c/c++ api version
 */
static PyObject* bob_learn_linear_version() {
  return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_LEARN_LINEAR_API_VERSION));
}


static PyObject* build_version_dictionary() {

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  if (!dict_set(retval, "Blitz++", BZ_VERSION)) return 0;
  if (!dict_steal(retval, "Boost", boost_version())) return 0;
  if (!dict_steal(retval, "Compiler", compiler_version())) return 0;
  if (!dict_steal(retval, "Python", python_version())) return 0;
  if (!dict_steal(retval, "NumPy", numpy_version())) return 0;
  if (!dict_steal(retval, "bob.blitz", bob_blitz_version())) return 0;
  if (!dict_steal(retval, "bob.core", bob_core_version())) return 0;
  if (!dict_steal(retval, "bob.io.base", bob_io_base_version())) return 0;
  if (!dict_steal(retval, "bob.sp", bob_sp_version())) return 0;
  if (!dict_steal(retval, "bob.math", bob_math_version())) return 0;
  if (!dict_steal(retval, "bob.learn.activation", bob_learn_activation_version())) return 0;
  if (!dict_steal(retval, "bob.learn.linear", bob_learn_linear_version())) return 0;
  if (!dict_steal(retval, "Bob", bob_core_version())) return 0;

  Py_INCREF(retval);
  return retval;
}

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"Information about software used to compile the C++ Bob API"
);

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
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m); ///< protects against early returns

  /* register version numbers and constants */
  if (PyModule_AddStringConstant(m, "module", BOB_EXT_MODULE_VERSION) < 0)
    return 0;

  PyObject* externals = build_version_dictionary();
  if (!externals) return 0;
  if (PyModule_AddObject(m, "externals", externals) < 0) return 0;

  /* imports dependencies */
  if (import_bob_blitz() < 0) {
    PyErr_Print();
    PyErr_Format(PyExc_ImportError, "cannot import `%s'", BOB_EXT_MODULE_NAME);
    return 0;
  }

  Py_INCREF(m);
  return m;

}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}

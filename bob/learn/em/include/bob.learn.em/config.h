/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Thu Aug 21 20:49:42 CEST 2014
 *
 * @brief General directives for all modules in bob.learn.em
 */

#ifndef BOB_LEARN_EM_CONFIG_H
#define BOB_LEARN_EM_CONFIG_H

/* Macros that define versions and important names */
#define BOB_LEARN_EM_API_VERSION 0x0200

#ifdef BOB_IMPORT_VERSION

  /***************************************
  * Here we define some functions that should be used to build version dictionaries in the version.cpp file
  * There will be a compiler warning, when these functions are not used, so use them!
  ***************************************/

  #include <Python.h>
  #include <boost/preprocessor/stringize.hpp>

  /**
   * bob.learn.em c/c++ api version
   */
  static PyObject* bob_learn_em_version() {
    return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_LEARN_EM_API_VERSION));
  }

#endif // BOB_IMPORT_VERSION

#endif /* BOB_LEARN_EM_CONFIG_H */

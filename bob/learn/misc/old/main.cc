/**
 * @author André Anjos <andre.anjos@idiap.ch>
 * @date Tue Jan 18 17:07:26 2011 +0100
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif

#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>
#include <bob.core/random_api.h>

#include "ndarray.h"

/** extra bindings required for compatibility **/
void bind_core_tinyvector();
void bind_core_ndarray_numpy();
void bind_core_bz_numpy();

/** machine bindings **/
void bind_machine_base();
void bind_machine_bic();
void bind_machine_gaussian();
void bind_machine_gmm();
void bind_machine_kmeans();
void bind_machine_linear_scoring();
void bind_machine_ztnorm();
void bind_machine_jfa();
void bind_machine_ivector();
void bind_machine_plda();
void bind_machine_wiener();

/** trainer bindings **/
void bind_trainer_gmm();
void bind_trainer_kmeans();
void bind_trainer_jfa();
void bind_trainer_ivector();
void bind_trainer_plda();
void bind_trainer_wiener();
void bind_trainer_empca();
void bind_trainer_bic();

BOOST_PYTHON_MODULE(_old_library) {

  if (import_bob_blitz() < 0) {
    PyErr_Print();
    PyErr_Format(PyExc_ImportError, "cannot import `bob.blitz'");
    return;
  }

  if (import_bob_core_random() < 0) {
    PyErr_Print();
    PyErr_Format(PyExc_ImportError, "cannot import `bob.core.random'");
    return;
  }

  if (import_bob_io_base() < 0) {
    PyErr_Print();
    PyErr_Format(PyExc_ImportError, "cannot import `bob.io.base'");
    return;
  }

  boost::python::docstring_options docopt(true, true, false);

  bob::python::setup_python("miscelaneous machines and trainers not yet ported into the new framework");

  /** extra bindings required for compatibility **/
  bind_core_tinyvector();
  bind_core_ndarray_numpy();
  bind_core_bz_numpy();

  /** machine bindings **/
  bind_machine_base();
  bind_machine_bic();
  bind_machine_gaussian();
  bind_machine_gmm();
  bind_machine_kmeans();
  bind_machine_linear_scoring();
  bind_machine_ztnorm();
  bind_machine_jfa();
  bind_machine_ivector();
  bind_machine_plda();
  bind_machine_wiener();

  /** trainer bindings **/
  bind_trainer_gmm();
  bind_trainer_kmeans();
  bind_trainer_jfa();
  bind_trainer_ivector();
  bind_trainer_plda();
  bind_trainer_wiener();
  bind_trainer_empca();
  bind_trainer_bic();

}

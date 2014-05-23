/**
 * @author André Anjos <andre.anjos@idiap.ch>
 * @date Tue Jan 18 17:07:26 2011 +0100
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "ndarray.h"

/** extra bindings required for compatibility **/
void bind_core_tinyvector();
void bind_core_ndarray_numpy();
void bind_core_bz_numpy();
void bind_ip_gabor_wavelet_transform();

/** machine bindings **/
void bind_machine_base();
void bind_machine_bic();
void bind_machine_gabor();
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

BOOST_PYTHON_MODULE(_library) {

  boost::python::docstring_options docopt(true, true, false);

  bob::python::setup_python("miscelaneous machines and trainers not yet ported into the new framework");

  /** extra bindings required for compatibility **/
  bind_core_tinyvector();
  bind_core_ndarray_numpy();
  bind_core_bz_numpy();
  bind_ip_gabor_wavelet_transform();

  /** machine bindings **/
  bind_machine_base();
  bind_machine_bic();
  bind_machine_gabor();
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

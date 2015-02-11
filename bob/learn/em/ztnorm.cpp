/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Sat 31 Jan 02:46:48 2015
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/*** zt_norm ***/
static auto zt_norm = bob::extension::FunctionDoc(
  "ztnorm",
  "",
  0,
  true
)
.add_prototype("rawscores_probes_vs_models,rawscores_zprobes_vs_models,rawscores_probes_vs_tmodels,rawscores_zprobes_vs_tmodels,mask_zprobes_vs_tmodels_istruetrial", "output")
.add_parameter("rawscores_probes_vs_models", "array_like <float, 2D>", "")
.add_parameter("rawscores_zprobes_vs_models", "array_like <float, 2D>", "")
.add_parameter("rawscores_probes_vs_tmodels", "array_like <float, 2D>", "")
.add_parameter("rawscores_zprobes_vs_tmodels", "array_like <float, 2D>", "")
.add_parameter("mask_zprobes_vs_tmodels_istruetrial", "array_like <float, 2D>", "")
.add_return("output","array_like <float, 2D>","");
static PyObject* PyBobLearnEM_ztNorm(PyObject*, PyObject* args, PyObject* kwargs) {

  char** kwlist = zt_norm.kwlist(0);
  
  PyBlitzArrayObject *rawscores_probes_vs_models_o, *rawscores_zprobes_vs_models_o, *rawscores_probes_vs_tmodels_o, 
  *rawscores_zprobes_vs_tmodels_o, *mask_zprobes_vs_tmodels_istruetrial_o;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&O&|O&", kwlist, &PyBlitzArray_Converter, &rawscores_probes_vs_models_o,
                                                                       &PyBlitzArray_Converter, &rawscores_zprobes_vs_models_o,
                                                                       &PyBlitzArray_Converter, &rawscores_probes_vs_tmodels_o,
                                                                       &PyBlitzArray_Converter, &rawscores_zprobes_vs_tmodels_o,
                                                                       &PyBlitzArray_Converter, &mask_zprobes_vs_tmodels_istruetrial_o)){
    zt_norm.print_usage();
    return 0;
  }

  // get the number of command line arguments
  auto rawscores_probes_vs_models_          = make_safe(rawscores_probes_vs_models_o);
  auto rawscores_zprobes_vs_models_         = make_safe(rawscores_zprobes_vs_models_o);
  auto rawscores_probes_vs_tmodels_         = make_safe(rawscores_probes_vs_tmodels_o);
  auto rawscores_zprobes_vs_tmodels_        = make_safe(rawscores_zprobes_vs_tmodels_o);
  //auto mask_zprobes_vs_tmodels_istruetrial_ = make_safe(mask_zprobes_vs_tmodels_istruetrial_o);

  blitz::Array<double,2>  rawscores_probes_vs_models = *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o);
  blitz::Array<double,2> normalized_scores = blitz::Array<double,2>(rawscores_probes_vs_models.extent(0), rawscores_probes_vs_models.extent(1));

  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if(nargs==4)
    bob::learn::em::ztNorm(*PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o),
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_models_o),
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_tmodels_o),
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_tmodels_o),
                             normalized_scores);
  else
    bob::learn::em::ztNorm(*PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o), 
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_models_o), 
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_tmodels_o), 
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_tmodels_o), 
                             *PyBlitzArrayCxx_AsBlitz<bool,2>(mask_zprobes_vs_tmodels_istruetrial_o),
                             normalized_scores);

  return PyBlitzArrayCxx_AsConstNumpy(normalized_scores);
}



/*** t_norm ***/
static auto t_norm = bob::extension::FunctionDoc(
  "tnorm",
  "",
  0,
  true
)
.add_prototype("rawscores_probes_vs_models,rawscores_probes_vs_tmodels", "output")
.add_parameter("rawscores_probes_vs_models", "array_like <float, 2D>", "")
.add_parameter("rawscores_probes_vs_tmodels", "array_like <float, 2D>", "")
.add_return("output","array_like <float, 2D>","");
static PyObject* PyBobLearnEM_tNorm(PyObject*, PyObject* args, PyObject* kwargs) {

  char** kwlist = zt_norm.kwlist(0);
  
  PyBlitzArrayObject *rawscores_probes_vs_models_o, *rawscores_probes_vs_tmodels_o;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&", kwlist, &PyBlitzArray_Converter, &rawscores_probes_vs_models_o,
                                                                       &PyBlitzArray_Converter, &rawscores_probes_vs_tmodels_o)){
    zt_norm.print_usage();
    return 0;
  }
  
  auto rawscores_probes_vs_models_          = make_safe(rawscores_probes_vs_models_o);
  auto rawscores_probes_vs_tmodels_         = make_safe(rawscores_probes_vs_tmodels_o);

  blitz::Array<double,2>  rawscores_probes_vs_models = *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o);
  blitz::Array<double,2> normalized_scores = blitz::Array<double,2>(rawscores_probes_vs_models.extent(0), rawscores_probes_vs_models.extent(1));

  bob::learn::em::tNorm(*PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o), 
                           *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_tmodels_o),
                           normalized_scores);

  return PyBlitzArrayCxx_AsConstNumpy(normalized_scores);
}


/*** z_norm ***/
static auto z_norm = bob::extension::FunctionDoc(
  "znorm",
  "",
  0,
  true
)
.add_prototype("rawscores_probes_vs_models,rawscores_zprobes_vs_models", "output")
.add_parameter("rawscores_probes_vs_models", "array_like <float, 2D>", "")
.add_parameter("rawscores_zprobes_vs_models", "array_like <float, 2D>", "")
.add_return("output","array_like <float, 2D>","");
static PyObject* PyBobLearnEM_zNorm(PyObject*, PyObject* args, PyObject* kwargs) {

  char** kwlist = zt_norm.kwlist(0);
  
  PyBlitzArrayObject *rawscores_probes_vs_models_o, *rawscores_zprobes_vs_models_o;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&", kwlist, &PyBlitzArray_Converter, &rawscores_probes_vs_models_o,
                                                                       &PyBlitzArray_Converter, &rawscores_zprobes_vs_models_o)){
    zt_norm.print_usage();
    return 0;
  }
  
  auto rawscores_probes_vs_models_          = make_safe(rawscores_probes_vs_models_o);
  auto rawscores_zprobes_vs_models_         = make_safe(rawscores_zprobes_vs_models_o);

  blitz::Array<double,2> rawscores_probes_vs_models = *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o);
  blitz::Array<double,2> normalized_scores          = blitz::Array<double,2>(rawscores_probes_vs_models.extent(0), rawscores_probes_vs_models.extent(1));


  bob::learn::em::zNorm(*PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o), 
                           *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_models_o),
                           normalized_scores);

  return PyBlitzArrayCxx_AsConstNumpy(normalized_scores);
}


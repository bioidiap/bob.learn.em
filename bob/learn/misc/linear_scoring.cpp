/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Wed 05 Feb 16:10:48 2015
 *
 * @brief Python API for bob::learn::em
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

/*Convert a PyObject to a a list of GMMStats*/
//template<class R, class P1, class P2>
static int extract_gmmstats_list(PyObject *list,
                             std::vector<boost::shared_ptr<const bob::learn::misc::GMMStats> >& training_data)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++){
  
    PyBobLearnMiscGMMStatsObject* stats;
    if (!PyArg_Parse(PyList_GetItem(list, i), "O!", &PyBobLearnMiscGMMStats_Type, &stats)){
      PyErr_Format(PyExc_RuntimeError, "Expected GMMStats objects");
      return -1;
    }
    training_data.push_back(stats->cxx);
  }
  return 0;
}

static int extract_gmmmachine_list(PyObject *list,
                             std::vector<boost::shared_ptr<const bob::learn::misc::GMMMachine> >& training_data)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++){
  
    PyBobLearnMiscGMMMachineObject* stats;
    if (!PyArg_Parse(PyList_GetItem(list, i), "O!", &PyBobLearnMiscGMMMachine_Type, &stats)){
      PyErr_Format(PyExc_RuntimeError, "Expected GMMMachine objects");
      return -1;
    }
    training_data.push_back(stats->cxx);
  }
  return 0;
}



/*Convert a PyObject to a list of blitz Array*/
template <int N>
int extract_array_list(PyObject* list, std::vector<blitz::Array<double,N> >& vec)
{

  if(list==0)
    return 0;

  for (int i=0; i<PyList_GET_SIZE(list); i++)
  {
    PyBlitzArrayObject* blitz_object; 
    if (!PyArg_Parse(PyList_GetItem(list, i), "O&", &PyBlitzArray_Converter, &blitz_object)){
      PyErr_Format(PyExc_RuntimeError, "Expected numpy array object");
      return -1;
    }
    auto blitz_object_ = make_safe(blitz_object);
    vec.push_back(*PyBlitzArrayCxx_AsBlitz<double,N>(blitz_object));
  }
  return 0;
}

/* converts PyObject to bool and returns false if object is NULL */
static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}


/*** linear_scoring ***/
static auto linear_scoring = bob::extension::FunctionDoc(
  "linear_scoring",
  "",
  0,
  true
)
.add_prototype("models, ubm, test_stats, test_channelOffset, frame_length_normalisation", "output")
.add_parameter("models", "", "")
.add_parameter("ubm", "", "")
.add_parameter("test_stats", "", "")
.add_parameter("test_channelOffset", "", "")
.add_parameter("frame_length_normalisation", "bool", "")
.add_return("output","array_like<float,1>","Score");
static PyObject* PyBobLearnMisc_linear_scoring(PyObject*, PyObject* args, PyObject* kwargs) {

  char** kwlist = linear_scoring.kwlist(0);
    
  //Cheking the number of arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

    //Read a list of GMM
  if((nargs >= 3) && (nargs<=5)){

    PyObject* gmm_list_o                 = 0;
    PyBobLearnMiscGMMMachineObject* ubm  = 0;
    PyObject* stats_list_o               = 0;
    PyObject* channel_offset_list_o      = 0;
    PyObject* frame_length_normalisation = Py_False;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!|O!O!", kwlist, &PyList_Type, &gmm_list_o,
                                                                       &PyBobLearnMiscGMMMachine_Type, &ubm,
                                                                       &PyList_Type, &stats_list_o,
                                                                       &PyList_Type, &channel_offset_list_o,
                                                                       &PyBool_Type, &frame_length_normalisation)){
      linear_scoring.print_usage(); 
      Py_RETURN_NONE;
    }

    std::vector<boost::shared_ptr<const bob::learn::misc::GMMStats> > stats_list;
    if(extract_gmmstats_list(stats_list_o ,stats_list)!=0)
      Py_RETURN_NONE;

    std::vector<boost::shared_ptr<const bob::learn::misc::GMMMachine> > gmm_list;
    if(extract_gmmmachine_list(gmm_list_o ,gmm_list)!=0)
      Py_RETURN_NONE;

    std::vector<blitz::Array<double,1> > channel_offset_list;
    if(extract_array_list(channel_offset_list_o ,channel_offset_list)!=0)
      Py_RETURN_NONE;

    blitz::Array<double, 2> scores = blitz::Array<double, 2>(gmm_list.size(), stats_list.size());
    if(channel_offset_list.size()==0)
      bob::learn::misc::linearScoring(gmm_list, *ubm->cxx, stats_list, f(frame_length_normalisation),scores);
    else
      bob::learn::misc::linearScoring(gmm_list, *ubm->cxx, stats_list, channel_offset_list, f(frame_length_normalisation),scores);

    return PyBlitzArrayCxx_AsConstNumpy(scores);
  }
  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - linear_scoring requires 5 or 6 arguments, but you provided %d (see help)", nargs);
    linear_scoring.print_usage();
    Py_RETURN_NONE;
  }
  /*
  
  
  PyBlitzArrayObject *rawscores_probes_vs_models_o, *rawscores_zprobes_vs_models_o, *rawscores_probes_vs_tmodels_o, 
  *rawscores_zprobes_vs_tmodels_o, *mask_zprobes_vs_tmodels_istruetrial_o;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&O&|O&", kwlist, &PyBlitzArray_Converter, &rawscores_probes_vs_models_o,
                                                                       &PyBlitzArray_Converter, &rawscores_zprobes_vs_models_o,
                                                                       &PyBlitzArray_Converter, &rawscores_probes_vs_tmodels_o,
                                                                       &PyBlitzArray_Converter, &rawscores_zprobes_vs_tmodels_o,
                                                                       &PyBlitzArray_Converter, &mask_zprobes_vs_tmodels_istruetrial_o)){
    zt_norm.print_usage();
    Py_RETURN_NONE;
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
    bob::learn::misc::ztNorm(*PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o),
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_models_o),
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_tmodels_o),
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_tmodels_o),
                             normalized_scores);
  else
    bob::learn::misc::ztNorm(*PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_models_o), 
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_models_o), 
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_probes_vs_tmodels_o), 
                             *PyBlitzArrayCxx_AsBlitz<double,2>(rawscores_zprobes_vs_tmodels_o), 
                             *PyBlitzArrayCxx_AsBlitz<bool,2>(mask_zprobes_vs_tmodels_istruetrial_o),
                             normalized_scores);

  return PyBlitzArrayCxx_AsConstNumpy(normalized_scores);
  */

}


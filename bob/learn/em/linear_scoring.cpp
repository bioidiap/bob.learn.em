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
                             std::vector<boost::shared_ptr<const bob::learn::em::GMMStats> >& training_data)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++){

    PyBobLearnEMGMMStatsObject* stats;
    if (!PyArg_Parse(PyList_GetItem(list, i), "O!", &PyBobLearnEMGMMStats_Type, &stats)){
      PyErr_Format(PyExc_RuntimeError, "Expected GMMStats objects");
      return -1;
    }
    training_data.push_back(stats->cxx);
  }
  return 0;
}

static int extract_gmmmachine_list(PyObject *list,
                             std::vector<boost::shared_ptr<const bob::learn::em::GMMMachine> >& training_data)
{
  for (int i=0; i<PyList_GET_SIZE(list); i++){

    PyBobLearnEMGMMMachineObject* stats;
    if (!PyArg_Parse(PyList_GetItem(list, i), "O!", &PyBobLearnEMGMMMachine_Type, &stats)){
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
bob::extension::FunctionDoc linear_scoring1 = bob::extension::FunctionDoc(
  "linear_scoring",
  "",
  0,
  true
)
.add_prototype("models, ubm, test_stats, test_channelOffset, frame_length_normalisation", "output")
.add_parameter("models", "[:py:class:`bob.learn.em.GMMMachine`]", "")
.add_parameter("ubm", ":py:class:`bob.learn.em.GMMMachine`", "")
.add_parameter("test_stats", "[:py:class:`bob.learn.em.GMMStats`]", "")
.add_parameter("test_channelOffset", "[array_like<float,1>]", "")
.add_parameter("frame_length_normalisation", "bool", "")
.add_return("output","array_like<float,1>","Score");


bob::extension::FunctionDoc linear_scoring2 = bob::extension::FunctionDoc(
  "linear_scoring",
  "",
  0,
  true
)
.add_prototype("models, ubm_mean, ubm_variance, test_stats, test_channelOffset, frame_length_normalisation", "output")
.add_parameter("models", "list(array_like<float,1>)", "")
.add_parameter("ubm_mean", "list(array_like<float,1>)", "")
.add_parameter("ubm_variance", "list(array_like<float,1>)", "")
.add_parameter("test_stats", "list(:py:class:`bob.learn.em.GMMStats`)", "")
.add_parameter("test_channelOffset", "list(array_like<float,1>)", "")
.add_parameter("frame_length_normalisation", "bool", "")
.add_return("output","array_like<float,1>","Score");



bob::extension::FunctionDoc linear_scoring3 = bob::extension::FunctionDoc(
  "linear_scoring",
  "",
  0,
  true
)
.add_prototype("model, ubm_mean, ubm_variance, test_stats, test_channelOffset, frame_length_normalisation", "output")
.add_parameter("model", "array_like<float,1>", "")
.add_parameter("ubm_mean", "array_like<float,1>", "")
.add_parameter("ubm_variance", "array_like<float,1>", "")
.add_parameter("test_stats", ":py:class:`bob.learn.em.GMMStats`", "")
.add_parameter("test_channelOffset", "array_like<float,1>", "")
.add_parameter("frame_length_normalisation", "bool", "")
.add_return("output","array_like<float,1>","Score");

PyObject* PyBobLearnEM_linear_scoring(PyObject*, PyObject* args, PyObject* kwargs) {

  //Cheking the number of arguments
  int nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  //Reading the first input argument
  PyObject* arg = 0;
  if (PyTuple_Size(args))
    arg = PyTuple_GET_ITEM(args, 0);
  else {
    PyObject* tmp = PyDict_Values(kwargs);
    auto tmp_ = make_safe(tmp);
    arg = PyList_GET_ITEM(tmp, 0);
  }

  //Checking the signature of the method (list of GMMMachine as input)
  if ((PyList_Check(arg)) && PyBobLearnEMGMMMachine_Check(PyList_GetItem(arg, 0)) && (nargs >= 3) && (nargs<=5) ){

    char** kwlist = linear_scoring1.kwlist(0);

    PyObject* gmm_list_o                 = 0;
    PyBobLearnEMGMMMachineObject* ubm  = 0;
    PyObject* stats_list_o               = 0;
    PyObject* channel_offset_list_o      = 0;
    PyObject* frame_length_normalisation = Py_False;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!|O!O!", kwlist, &PyList_Type, &gmm_list_o,
                                                                       &PyBobLearnEMGMMMachine_Type, &ubm,
                                                                       &PyList_Type, &stats_list_o,
                                                                       &PyList_Type, &channel_offset_list_o,
                                                                       &PyBool_Type, &frame_length_normalisation)){
      linear_scoring1.print_usage();
      return 0;
    }

    std::vector<boost::shared_ptr<const bob::learn::em::GMMStats> > stats_list;
    if(extract_gmmstats_list(stats_list_o ,stats_list)!=0)
      Py_RETURN_NONE;

    std::vector<boost::shared_ptr<const bob::learn::em::GMMMachine> > gmm_list;
    if(extract_gmmmachine_list(gmm_list_o ,gmm_list)!=0)
      Py_RETURN_NONE;

    std::vector<blitz::Array<double,1> > channel_offset_list;
    if(extract_array_list(channel_offset_list_o ,channel_offset_list)!=0)
      Py_RETURN_NONE;

    blitz::Array<double, 2> scores = blitz::Array<double, 2>(gmm_list.size(), stats_list.size());
    if(channel_offset_list.size()==0)
      bob::learn::em::linearScoring(gmm_list, *ubm->cxx, stats_list, f(frame_length_normalisation),scores);
    else
      bob::learn::em::linearScoring(gmm_list, *ubm->cxx, stats_list, channel_offset_list, f(frame_length_normalisation),scores);

    return PyBlitzArrayCxx_AsConstNumpy(scores);
  }

  //Checking the signature of the method (list of arrays as input
  else if ((PyList_Check(arg)) && PyArray_Check(PyList_GetItem(arg, 0)) && (nargs >= 4) && (nargs<=6) ){

    char** kwlist = linear_scoring2.kwlist(0);

    PyObject* model_supervector_list_o        = 0;
    PyBlitzArrayObject* ubm_means             = 0;
    PyBlitzArrayObject* ubm_variances         = 0;
    PyObject* stats_list_o                    = 0;
    PyObject* channel_offset_list_o           = 0;
    PyObject* frame_length_normalisation      = Py_False;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&O&O!|O!O!", kwlist, &PyList_Type, &model_supervector_list_o,
                                                                       &PyBlitzArray_Converter, &ubm_means,
                                                                       &PyBlitzArray_Converter, &ubm_variances,
                                                                       &PyList_Type, &stats_list_o,
                                                                       &PyList_Type, &channel_offset_list_o,
                                                                       &PyBool_Type, &frame_length_normalisation)){
      linear_scoring2.print_usage();
      return 0;
    }

    //protects acquired resources through this scope
    auto ubm_means_ = make_safe(ubm_means);
    auto ubm_variances_ = make_safe(ubm_variances);

    std::vector<blitz::Array<double,1> > model_supervector_list;
    if(extract_array_list(model_supervector_list_o ,model_supervector_list)!=0)
      Py_RETURN_NONE;

    std::vector<boost::shared_ptr<const bob::learn::em::GMMStats> > stats_list;
    if(extract_gmmstats_list(stats_list_o ,stats_list)!=0)
      Py_RETURN_NONE;

    std::vector<blitz::Array<double,1> > channel_offset_list;
    if(extract_array_list(channel_offset_list_o ,channel_offset_list)!=0)
      Py_RETURN_NONE;

    blitz::Array<double, 2> scores = blitz::Array<double, 2>(model_supervector_list.size(), stats_list.size());
    if(channel_offset_list.size()==0)
      bob::learn::em::linearScoring(model_supervector_list, *PyBlitzArrayCxx_AsBlitz<double,1>(ubm_means),*PyBlitzArrayCxx_AsBlitz<double,1>(ubm_variances), stats_list, f(frame_length_normalisation),scores);
    else
      bob::learn::em::linearScoring(model_supervector_list, *PyBlitzArrayCxx_AsBlitz<double,1>(ubm_means),*PyBlitzArrayCxx_AsBlitz<double,1>(ubm_variances), stats_list, channel_offset_list, f(frame_length_normalisation),scores);

    return PyBlitzArrayCxx_AsConstNumpy(scores);

  }

  //Checking the signature of the method (list of arrays as input
  else if (PyArray_Check(arg) && (nargs >= 5) && (nargs<=6) ){

    char** kwlist = linear_scoring3.kwlist(0);

    PyBlitzArrayObject* model                 = 0;
    PyBlitzArrayObject* ubm_means             = 0;
    PyBlitzArrayObject* ubm_variances         = 0;
    PyBobLearnEMGMMStatsObject* stats       = 0;
    PyBlitzArrayObject* channel_offset        = 0;
    PyObject* frame_length_normalisation      = Py_False;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&O&O!O&|O!", kwlist, &PyBlitzArray_Converter, &model,
                                                                       &PyBlitzArray_Converter, &ubm_means,
                                                                       &PyBlitzArray_Converter, &ubm_variances,
                                                                       &PyBobLearnEMGMMStats_Type, &stats,
                                                                       &PyBlitzArray_Converter, &channel_offset,
                                                                       &PyBool_Type, &frame_length_normalisation)){
      linear_scoring3.print_usage();
      return 0;
    }

    //protects acquired resources through this scope
    auto model_ = make_safe(model);
    auto ubm_means_ = make_safe(ubm_means);
    auto ubm_variances_ = make_safe(ubm_variances);
    auto channel_offset_ = make_safe(channel_offset);

    double score = bob::learn::em::linearScoring(*PyBlitzArrayCxx_AsBlitz<double,1>(model), *PyBlitzArrayCxx_AsBlitz<double,1>(ubm_means),*PyBlitzArrayCxx_AsBlitz<double,1>(ubm_variances), *stats->cxx, *PyBlitzArrayCxx_AsBlitz<double,1>(channel_offset), f(frame_length_normalisation));

    return Py_BuildValue("d",score);
  }


  else{
    PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - linear_scoring requires 5 or 6 arguments, but you provided %d (see help)", nargs);
    linear_scoring1.print_usage();
    linear_scoring2.print_usage();
    linear_scoring3.print_usage();
    return 0;
  }

}

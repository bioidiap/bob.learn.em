/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue Jul 19 12:16:17 2011 +0200
 *
 * @brief Python bindings to Joint Factor Analysis trainers
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "ndarray.h"
#include <boost/python/stl_iterator.hpp>
#include <bob.learn.misc/JFATrainer.h>

using namespace boost::python;

template <int N>
static object vector_as_list(const std::vector<blitz::Array<double,N> >& vec)
{
  list retval;
  for(size_t k=0; k<vec.size(); ++k)
    retval.append(vec[k]);
  return tuple(retval);
}

static void extract_GMMStats(object data,
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > >& training_data)
{
  stl_input_iterator<object> dbegin(data), dend;
  std::vector<object> vvdata(dbegin, dend);
  for (size_t i=0; i<vvdata.size(); ++i)
  {
    stl_input_iterator<boost::shared_ptr<bob::learn::misc::GMMStats> > dlbegin(vvdata[i]), dlend;
    training_data.push_back(std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> >(dlbegin, dlend));
  }
}

static void isv_train(bob::learn::misc::ISVTrainer& t, bob::learn::misc::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the train function
  t.train(m, training_data);
}

static void isv_initialize(bob::learn::misc::ISVTrainer& t, bob::learn::misc::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the initialize function
  t.initialize(m, training_data);
}

static void isv_estep(bob::learn::misc::ISVTrainer& t, bob::learn::misc::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the E-Step function
  t.eStep(m, training_data);
}

static void isv_mstep(bob::learn::misc::ISVTrainer& t, bob::learn::misc::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the M-Step function
  t.mStep(m, training_data);
}

static void isv_finalize(bob::learn::misc::ISVTrainer& t, bob::learn::misc::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the finalization function
  t.finalize(m, training_data);
}

static void isv_enrol(bob::learn::misc::ISVTrainer& t, bob::learn::misc::ISVMachine& m, object data, const size_t n_iter)
{
  stl_input_iterator<boost::shared_ptr<bob::learn::misc::GMMStats> > dlbegin(data), dlend;
  std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > vdata(dlbegin, dlend);
  // Calls the enrol function
  t.enrol(m, vdata, n_iter);
}

static object isv_get_x(const bob::learn::misc::ISVTrainer& t)
{
  return vector_as_list(t.getX());
}

static object isv_get_z(const bob::learn::misc::ISVTrainer& t)
{
  return vector_as_list(t.getZ());
}

static void isv_set_x(bob::learn::misc::ISVTrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,2>());
  t.setX(vdata_ref);
}

static void isv_set_z(bob::learn::misc::ISVTrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,1>());
  t.setZ(vdata_ref);
}


static void isv_set_accUA1(bob::learn::misc::ISVTrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccUA1(acc.bz<double,3>());
}

static void isv_set_accUA2(bob::learn::misc::ISVTrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccUA2(acc.bz<double,2>());
}



static void jfa_train(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the train function
  t.train(m, training_data);
}

static void jfa_initialize(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the initialize function
  t.initialize(m, training_data);
}

static void jfa_estep1(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the E-Step function
  t.eStep1(m, training_data);
}

static void jfa_mstep1(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the M-Step function
  t.mStep1(m, training_data);
}

static void jfa_finalize1(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the finalization function
  t.finalize1(m, training_data);
}

static void jfa_estep2(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the E-Step function
  t.eStep2(m, training_data);
}

static void jfa_mstep2(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the M-Step function
  t.mStep2(m, training_data);
}

static void jfa_finalize2(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the finalization function
  t.finalize2(m, training_data);
}

static void jfa_estep3(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the E-Step function
  t.eStep3(m, training_data);
}

static void jfa_mstep3(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the M-Step function
  t.mStep3(m, training_data);
}

static void jfa_finalize3(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the finalization function
  t.finalize3(m, training_data);
}

static void jfa_train_loop(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the main loop function
  t.train_loop(m, training_data);
}

static void jfa_enrol(bob::learn::misc::JFATrainer& t, bob::learn::misc::JFAMachine& m, object data, const size_t n_iter)
{
  stl_input_iterator<boost::shared_ptr<bob::learn::misc::GMMStats> > dlbegin(data), dlend;
  std::vector<boost::shared_ptr<bob::learn::misc::GMMStats> > vdata(dlbegin, dlend);
  // Calls the enrol function
  t.enrol(m, vdata, n_iter);
}

static object jfa_get_x(const bob::learn::misc::JFATrainer& t)
{
  return vector_as_list(t.getX());
}

static object jfa_get_y(const bob::learn::misc::JFATrainer& t)
{
  return vector_as_list(t.getY());
}

static object jfa_get_z(const bob::learn::misc::JFATrainer& t)
{
  return vector_as_list(t.getZ());
}

static void jfa_set_x(bob::learn::misc::JFATrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,2>());
  t.setX(vdata_ref);
}

static void jfa_set_y(bob::learn::misc::JFATrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,1>());
  t.setY(vdata_ref);
}

static void jfa_set_z(bob::learn::misc::JFATrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,1>());
  t.setZ(vdata_ref);
}


static void jfa_set_accUA1(bob::learn::misc::JFATrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccUA1(acc.bz<double,3>());
}

static void jfa_set_accUA2(bob::learn::misc::JFATrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccUA2(acc.bz<double,2>());
}

static void jfa_set_accVA1(bob::learn::misc::JFATrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccVA1(acc.bz<double,3>());
}

static void jfa_set_accVA2(bob::learn::misc::JFATrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccVA2(acc.bz<double,2>());
}

static void jfa_set_accDA1(bob::learn::misc::JFATrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccDA1(acc.bz<double,1>());
}

static void jfa_set_accDA2(bob::learn::misc::JFATrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccDA2(acc.bz<double,1>());
}



// include the random API of bob.core
#include <bob.core/random_api.h>
static boost::python::object isv_getRng(bob::learn::misc::ISVTrainer& self){
  // create new object
  PyObject* o = PyBoostMt19937_Type.tp_alloc(&PyBoostMt19937_Type,0);
  reinterpret_cast<PyBoostMt19937Object*>(o)->rng = self.getRng().get();
  return boost::python::object(boost::python::handle<>(boost::python::borrowed(o)));
}
static boost::python::object jfa_getRng(bob::learn::misc::JFATrainer& self){
  // create new object
  PyObject* o = PyBoostMt19937_Type.tp_alloc(&PyBoostMt19937_Type,0);
  reinterpret_cast<PyBoostMt19937Object*>(o)->rng = self.getRng().get();
  return boost::python::object(boost::python::handle<>(boost::python::borrowed(o)));
}

#include <boost/make_shared.hpp>
static void isv_setRng(bob::learn::misc::ISVTrainer& self, boost::python::object rng){
  if (!PyBoostMt19937_Check(rng.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.core.random.mt19937 object");
  PyBoostMt19937Object* o = reinterpret_cast<PyBoostMt19937Object*>(rng.ptr());
  self.setRng(boost::make_shared<boost::mt19937>(*o->rng));
}
static void jfa_setRng(bob::learn::misc::JFATrainer& self, boost::python::object rng){
  if (!PyBoostMt19937_Check(rng.ptr())) PYTHON_ERROR(TypeError, "Would have expected a bob.core.random.mt19937 object");
  PyBoostMt19937Object* o = reinterpret_cast<PyBoostMt19937Object*>(rng.ptr());
  self.setRng(boost::make_shared<boost::mt19937>(*o->rng));
}


void bind_trainer_jfa()
{
  class_<bob::learn::misc::ISVTrainer, boost::noncopyable >("ISVTrainer", "A trainer for Inter-session Variability Modelling (ISV). \n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", init<optional<const size_t, const double> >((arg("self"), arg("max_iterations")=10, arg("relevance_factor")=4.),"Initializes a new ISVTrainer."))
    .def(init<const bob::learn::misc::ISVTrainer&>((arg("self"), arg("other")), "Copy constructs an ISVTrainer"))
    .add_property("max_iterations", &bob::learn::misc::ISVTrainer::getMaxIterations, &bob::learn::misc::ISVTrainer::setMaxIterations, "Max iterations")
    .add_property("rng", &isv_getRng, &isv_setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .add_property("__X__", &isv_get_x, &isv_set_x)
    .add_property("__Z__", &isv_get_z, &isv_set_z)
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::ISVTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this ISVTrainer with the 'other' one to be approximately the same.")
    .def("train", &isv_train, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the training procedure.")
    .def("initialize", &isv_initialize, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the initialization procedure.")
    .def("e_step", &isv_estep, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the e-step procedure.")
    .def("m_step", &isv_mstep, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the m-step procedure.")
    .def("finalize", &isv_finalize, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the finalization procedure.")
    .def("enrol", &isv_enrol, (arg("self"), arg("isv_machine"), arg("gmm_stats"), arg("n_iter")), "Call the enrolment procedure.")
    .add_property("acc_u_a1", make_function(&bob::learn::misc::ISVTrainer::getAccUA1, return_value_policy<copy_const_reference>()), &isv_set_accUA1, "Accumulator updated during the E-step")
    .add_property("acc_u_a2", make_function(&bob::learn::misc::ISVTrainer::getAccUA2, return_value_policy<copy_const_reference>()), &isv_set_accUA2, "Accumulator updated during the E-step")
  ;

  class_<bob::learn::misc::JFATrainer, boost::noncopyable >("JFATrainer", "A trainer for Joint Factor Analysis (JFA).\n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", init<optional<const size_t> >((arg("self"), arg("max_iterations")=10),"Initializes a new JFATrainer."))
    .def(init<const bob::learn::misc::JFATrainer&>((arg("self"), arg("other")), "Copy constructs an JFATrainer"))
    .add_property("max_iterations", &bob::learn::misc::JFATrainer::getMaxIterations, &bob::learn::misc::JFATrainer::setMaxIterations, "Max iterations")
    .add_property("rng", &jfa_getRng, &jfa_setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .add_property("__X__", &jfa_get_x, &jfa_set_x)
    .add_property("__Y__", &jfa_get_y, &jfa_set_y)
    .add_property("__Z__", &jfa_get_z, &jfa_set_z)
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::learn::misc::JFATrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this JFATrainer with the 'other' one to be approximately the same.")
    .def("train", &jfa_train, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the training procedure.")
    .def("initialize", &jfa_initialize, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the initialization procedure.")
    .def("train_loop", &jfa_train_loop, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the training procedure (without the initialization). This will train the three subspaces U, V and d.")
    .def("e_step1", &jfa_estep1, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 1st e-step procedure (for the V subspace).")
    .def("m_step1", &jfa_mstep1, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 1st m-step procedure (for the V subspace).")
    .def("finalize1", &jfa_finalize1, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 1st finalization procedure (for the V subspace).")
    .def("e_step2", &jfa_estep2, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 2nd e-step procedure (for the U subspace).")
    .def("m_step2", &jfa_mstep2, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 2nd m-step procedure (for the U subspace).")
    .def("finalize2", &jfa_finalize2, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 2nd finalization procedure (for the U subspace).")
    .def("e_step3", &jfa_estep3, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 3rd e-step procedure (for the d subspace).")
    .def("m_step3", &jfa_mstep3, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 3rd m-step procedure (for the d subspace).")
    .def("finalize3", &jfa_finalize3, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 3rd finalization procedure (for the d subspace).")
    .def("enrol", &jfa_enrol, (arg("self"), arg("jfa_machine"), arg("gmm_stats"), arg("n_iter")), "Call the enrolment procedure.")
    .add_property("acc_v_a1", make_function(&bob::learn::misc::JFATrainer::getAccVA1, return_value_policy<copy_const_reference>()), &jfa_set_accVA1, "Accumulator updated during the E-step")
    .add_property("acc_v_a2", make_function(&bob::learn::misc::JFATrainer::getAccVA2, return_value_policy<copy_const_reference>()), &jfa_set_accVA2, "Accumulator updated during the E-step")
    .add_property("acc_u_a1", make_function(&bob::learn::misc::JFATrainer::getAccUA1, return_value_policy<copy_const_reference>()), &jfa_set_accUA1, "Accumulator updated during the E-step")
    .add_property("acc_u_a2", make_function(&bob::learn::misc::JFATrainer::getAccUA2, return_value_policy<copy_const_reference>()), &jfa_set_accUA2, "Accumulator updated during the E-step")
    .add_property("acc_d_a1", make_function(&bob::learn::misc::JFATrainer::getAccDA1, return_value_policy<copy_const_reference>()), &jfa_set_accDA1, "Accumulator updated during the E-step")
    .add_property("acc_d_a2", make_function(&bob::learn::misc::JFATrainer::getAccDA2, return_value_policy<copy_const_reference>()), &jfa_set_accDA2, "Accumulator updated during the E-step")
  ;
}

/**
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 * @date Fri Nov 21 10:31:25 CET 2014
 *
 * @brief Header file for bindings to bob::learn::em
 */

#ifndef BOB_LEARN_EM_MAIN_H
#define BOB_LEARN_EM_MAIN_H

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/random_api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#define BOB_LEARN_EM_MODULE
#include <bob.learn.misc/api.h>

#include <bob.learn.misc/Gaussian.h>
#include <bob.learn.misc/GMMStats.h>
#include <bob.learn.misc/GMMMachine.h>
#include <bob.learn.misc/KMeansMachine.h>
#include <bob.learn.misc/KMeansTrainer.h>

#include <bob.learn.misc/GMMBaseTrainer.h>
#include <bob.learn.misc/ML_GMMTrainer.h>


#if PY_VERSION_HEX >= 0x03000000
#define PyInt_Check PyLong_Check
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyString_Check PyUnicode_Check
#define PyString_AS_STRING(x) PyBytes_AS_STRING(make_safe(PyUnicode_AsUTF8String(x)).get())
#endif

#define TRY try{

#define CATCH(message,ret) }\
  catch (std::exception& e) {\
    PyErr_SetString(PyExc_RuntimeError, e.what());\
    return ret;\
  } \
  catch (...) {\
    PyErr_Format(PyExc_RuntimeError, "%s " message ": unknown exception caught", Py_TYPE(self)->tp_name);\
    return ret;\
  }

#define CATCH_(message, ret) }\
  catch (std::exception& e) {\
    PyErr_SetString(PyExc_RuntimeError, e.what());\
    return ret;\
  } \
  catch (...) {\
    PyErr_Format(PyExc_RuntimeError, message ": unknown exception caught");\
    return ret;\
  }

static inline char* c(const char* o){return const_cast<char*>(o);}  /* converts const char* to char* */

/// inserts the given key, value pair into the given dictionaries
static inline int insert_item_string(PyObject* dict, PyObject* entries, const char* key, Py_ssize_t value){
  auto v = make_safe(Py_BuildValue("n", value));
  if (PyDict_SetItemString(dict, key, v.get()) < 0) return -1;
  return PyDict_SetItemString(entries, key, v.get());
}

// Gaussian
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::Gaussian> cxx;
} PyBobLearnMiscGaussianObject;

extern PyTypeObject PyBobLearnMiscGaussian_Type;
bool init_BobLearnMiscGaussian(PyObject* module);
int PyBobLearnMiscGaussian_Check(PyObject* o);


// GMMStats
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::GMMStats> cxx;
} PyBobLearnMiscGMMStatsObject;

extern PyTypeObject PyBobLearnMiscGMMStats_Type;
bool init_BobLearnMiscGMMStats(PyObject* module);
int PyBobLearnMiscGMMStats_Check(PyObject* o);


// GMMMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::GMMMachine> cxx;
} PyBobLearnMiscGMMMachineObject;

extern PyTypeObject PyBobLearnMiscGMMMachine_Type;
bool init_BobLearnMiscGMMMachine(PyObject* module);
int PyBobLearnMiscGMMMachine_Check(PyObject* o);


// KMeansMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::KMeansMachine> cxx;
} PyBobLearnMiscKMeansMachineObject;

extern PyTypeObject PyBobLearnMiscKMeansMachine_Type;
bool init_BobLearnMiscKMeansMachine(PyObject* module);
int PyBobLearnMiscKMeansMachine_Check(PyObject* o);


// KMeansTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::KMeansTrainer> cxx;
} PyBobLearnMiscKMeansTrainerObject;

extern PyTypeObject PyBobLearnMiscKMeansTrainer_Type;
bool init_BobLearnMiscKMeansTrainer(PyObject* module);
int PyBobLearnMiscKMeansTrainer_Check(PyObject* o);


// GMMBaseTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::GMMBaseTrainer> cxx;
} PyBobLearnMiscGMMBaseTrainerObject;

extern PyTypeObject PyBobLearnMiscGMMBaseTrainer_Type;
bool init_BobLearnMiscGMMBaseTrainer(PyObject* module);
int PyBobLearnMiscGMMBaseTrainer_Check(PyObject* o);



// ML_GMMTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::ML_GMMTrainer> cxx;
} PyBobLearnMiscMLGMMTrainerObject;

extern PyTypeObject PyBobLearnMiscMLGMMTrainer_Type;
bool init_BobLearnMiscMLGMMTrainer(PyObject* module);
int PyBobLearnMiscMLGMMTrainer_Check(PyObject* o);




#endif // BOB_LEARN_EM_MAIN_H

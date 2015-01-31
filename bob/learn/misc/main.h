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
#include <bob.learn.misc/MAP_GMMTrainer.h>

#include <bob.learn.misc/JFABase.h>
#include <bob.learn.misc/ISVBase.h>

#include <bob.learn.misc/JFAMachine.h>
#include <bob.learn.misc/ISVMachine.h>
#include <bob.learn.misc/IVectorMachine.h>
#include <bob.learn.misc/PLDAMachine.h>
#include <bob.learn.misc/ZTNorm.h>

#include "ztnorm.cpp"


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


// MAP_GMMTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::MAP_GMMTrainer> cxx;
} PyBobLearnMiscMAPGMMTrainerObject;

extern PyTypeObject PyBobLearnMiscMAPGMMTrainer_Type;
bool init_BobLearnMiscMAPGMMTrainer(PyObject* module);
int PyBobLearnMiscMAPGMMTrainer_Check(PyObject* o);


// JFABase
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::JFABase> cxx;
} PyBobLearnMiscJFABaseObject;

extern PyTypeObject PyBobLearnMiscJFABase_Type;
bool init_BobLearnMiscJFABase(PyObject* module);
int PyBobLearnMiscJFABase_Check(PyObject* o);


// ISVBase
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::ISVBase> cxx;
} PyBobLearnMiscISVBaseObject;

extern PyTypeObject PyBobLearnMiscISVBase_Type;
bool init_BobLearnMiscISVBase(PyObject* module);
int PyBobLearnMiscISVBase_Check(PyObject* o);


// JFAMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::JFAMachine> cxx;
} PyBobLearnMiscJFAMachineObject;

extern PyTypeObject PyBobLearnMiscJFAMachine_Type;
bool init_BobLearnMiscJFAMachine(PyObject* module);
int PyBobLearnMiscJFAMachine_Check(PyObject* o);


// ISVMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::ISVMachine> cxx;
} PyBobLearnMiscISVMachineObject;

extern PyTypeObject PyBobLearnMiscISVMachine_Type;
bool init_BobLearnMiscISVMachine(PyObject* module);
int PyBobLearnMiscISVMachine_Check(PyObject* o);


// IVectorMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::IVectorMachine> cxx;
} PyBobLearnMiscIVectorMachineObject;

extern PyTypeObject PyBobLearnMiscIVectorMachine_Type;
bool init_BobLearnMiscIVectorMachine(PyObject* module);
int PyBobLearnMiscIVectorMachine_Check(PyObject* o);


// PLDABase
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::PLDABase> cxx;
} PyBobLearnMiscPLDABaseObject;

extern PyTypeObject PyBobLearnMiscPLDABase_Type;
bool init_BobLearnMiscPLDABase(PyObject* module);
int PyBobLearnMiscPLDABase_Check(PyObject* o);


// PLDAMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::misc::PLDAMachine> cxx;
} PyBobLearnMiscPLDAMachineObject;

extern PyTypeObject PyBobLearnMiscPLDAMachine_Type;
bool init_BobLearnMiscPLDAMachine(PyObject* module);
int PyBobLearnMiscPLDAMachine_Check(PyObject* o);



#endif // BOB_LEARN_EM_MAIN_H

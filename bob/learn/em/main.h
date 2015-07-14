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
#include <bob.core/api.h>
#include <bob.core/random_api.h>
#include <bob.io.base/api.h>
#include <bob.sp/api.h>
#include <bob.learn.activation/api.h>
#include <bob.learn.linear/api.h>

#include <bob.extension/documentation.h>

#define BOB_LEARN_EM_MODULE
#include <bob.learn.em/api.h>

#include <bob.learn.em/Gaussian.h>
#include <bob.learn.em/GMMStats.h>
#include <bob.learn.em/GMMMachine.h>
#include <bob.learn.em/KMeansMachine.h>

#include <bob.learn.em/KMeansTrainer.h>
//#include <bob.learn.em/GMMBaseTrainer.h>
#include <bob.learn.em/ML_GMMTrainer.h>
#include <bob.learn.em/MAP_GMMTrainer.h>

#include <bob.learn.em/JFABase.h>
#include <bob.learn.em/JFAMachine.h>
#include <bob.learn.em/JFATrainer.h>

#include <bob.learn.em/ISVBase.h>
#include <bob.learn.em/ISVMachine.h>
#include <bob.learn.em/ISVTrainer.h>


#include <bob.learn.em/IVectorMachine.h>
#include <bob.learn.em/IVectorTrainer.h>

#include <bob.learn.em/EMPCATrainer.h>

#include <bob.learn.em/PLDAMachine.h>
#include <bob.learn.em/PLDATrainer.h>

#include <bob.learn.em/ZTNorm.h>

/// inserts the given key, value pair into the given dictionaries
static inline int insert_item_string(PyObject* dict, PyObject* entries, const char* key, Py_ssize_t value){
  auto v = make_safe(Py_BuildValue("n", value));
  if (PyDict_SetItemString(dict, key, v.get()) < 0) return -1;
  return PyDict_SetItemString(entries, key, v.get());
}

// Gaussian
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::Gaussian> cxx;
} PyBobLearnEMGaussianObject;

extern PyTypeObject PyBobLearnEMGaussian_Type;
bool init_BobLearnEMGaussian(PyObject* module);
int PyBobLearnEMGaussian_Check(PyObject* o);


// GMMStats
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::GMMStats> cxx;
} PyBobLearnEMGMMStatsObject;

extern PyTypeObject PyBobLearnEMGMMStats_Type;
bool init_BobLearnEMGMMStats(PyObject* module);
int PyBobLearnEMGMMStats_Check(PyObject* o);


// GMMMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::GMMMachine> cxx;
} PyBobLearnEMGMMMachineObject;

extern PyTypeObject PyBobLearnEMGMMMachine_Type;
bool init_BobLearnEMGMMMachine(PyObject* module);
int PyBobLearnEMGMMMachine_Check(PyObject* o);


// KMeansMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::KMeansMachine> cxx;
} PyBobLearnEMKMeansMachineObject;

extern PyTypeObject PyBobLearnEMKMeansMachine_Type;
bool init_BobLearnEMKMeansMachine(PyObject* module);
int PyBobLearnEMKMeansMachine_Check(PyObject* o);


// KMeansTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::KMeansTrainer> cxx;
} PyBobLearnEMKMeansTrainerObject;

extern PyTypeObject PyBobLearnEMKMeansTrainer_Type;
bool init_BobLearnEMKMeansTrainer(PyObject* module);
int PyBobLearnEMKMeansTrainer_Check(PyObject* o);


// ML_GMMTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::ML_GMMTrainer> cxx;
} PyBobLearnEMMLGMMTrainerObject;

extern PyTypeObject PyBobLearnEMMLGMMTrainer_Type;
bool init_BobLearnEMMLGMMTrainer(PyObject* module);
int PyBobLearnEMMLGMMTrainer_Check(PyObject* o);


// MAP_GMMTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::MAP_GMMTrainer> cxx;
} PyBobLearnEMMAPGMMTrainerObject;

extern PyTypeObject PyBobLearnEMMAPGMMTrainer_Type;
bool init_BobLearnEMMAPGMMTrainer(PyObject* module);
int PyBobLearnEMMAPGMMTrainer_Check(PyObject* o);


// JFABase
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::JFABase> cxx;
} PyBobLearnEMJFABaseObject;

extern PyTypeObject PyBobLearnEMJFABase_Type;
bool init_BobLearnEMJFABase(PyObject* module);
int PyBobLearnEMJFABase_Check(PyObject* o);


// ISVBase
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::ISVBase> cxx;
} PyBobLearnEMISVBaseObject;

extern PyTypeObject PyBobLearnEMISVBase_Type;
bool init_BobLearnEMISVBase(PyObject* module);
int PyBobLearnEMISVBase_Check(PyObject* o);


// JFAMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::JFAMachine> cxx;
} PyBobLearnEMJFAMachineObject;

extern PyTypeObject PyBobLearnEMJFAMachine_Type;
bool init_BobLearnEMJFAMachine(PyObject* module);
int PyBobLearnEMJFAMachine_Check(PyObject* o);

// JFATrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::JFATrainer> cxx;
} PyBobLearnEMJFATrainerObject;


extern PyTypeObject PyBobLearnEMJFATrainer_Type;
bool init_BobLearnEMJFATrainer(PyObject* module);
int PyBobLearnEMJFATrainer_Check(PyObject* o);

// ISVMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::ISVMachine> cxx;
} PyBobLearnEMISVMachineObject;

extern PyTypeObject PyBobLearnEMISVMachine_Type;
bool init_BobLearnEMISVMachine(PyObject* module);
int PyBobLearnEMISVMachine_Check(PyObject* o);

// ISVTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::ISVTrainer> cxx;
} PyBobLearnEMISVTrainerObject;

extern PyTypeObject PyBobLearnEMISVTrainer_Type;
bool init_BobLearnEMISVTrainer(PyObject* module);
int PyBobLearnEMISVTrainer_Check(PyObject* o);

// IVectorMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::IVectorMachine> cxx;
} PyBobLearnEMIVectorMachineObject;

extern PyTypeObject PyBobLearnEMIVectorMachine_Type;
bool init_BobLearnEMIVectorMachine(PyObject* module);
int PyBobLearnEMIVectorMachine_Check(PyObject* o);


// IVectorTrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::IVectorTrainer> cxx;
} PyBobLearnEMIVectorTrainerObject;

extern PyTypeObject PyBobLearnEMIVectorTrainer_Type;
bool init_BobLearnEMIVectorTrainer(PyObject* module);
int PyBobLearnEMIVectorTrainer_Check(PyObject* o);


// PLDABase
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::PLDABase> cxx;
} PyBobLearnEMPLDABaseObject;

extern PyTypeObject PyBobLearnEMPLDABase_Type;
bool init_BobLearnEMPLDABase(PyObject* module);
int PyBobLearnEMPLDABase_Check(PyObject* o);


// PLDAMachine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::PLDAMachine> cxx;
} PyBobLearnEMPLDAMachineObject;

extern PyTypeObject PyBobLearnEMPLDAMachine_Type;
bool init_BobLearnEMPLDAMachine(PyObject* module);
int PyBobLearnEMPLDAMachine_Check(PyObject* o);


// PLDATrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::PLDATrainer> cxx;
} PyBobLearnEMPLDATrainerObject;

extern PyTypeObject PyBobLearnEMPLDATrainer_Type;
bool init_BobLearnEMPLDATrainer(PyObject* module);
int PyBobLearnEMPLDATrainer_Check(PyObject* o);



// EMPCATrainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::em::EMPCATrainer> cxx;
} PyBobLearnEMEMPCATrainerObject;

extern PyTypeObject PyBobLearnEMEMPCATrainer_Type;
bool init_BobLearnEMEMPCATrainer(PyObject* module);
int PyBobLearnEMEMPCATrainer_Check(PyObject* o);


//ZT Normalization
PyObject* PyBobLearnEM_ztNorm(PyObject*, PyObject* args, PyObject* kwargs);
extern bob::extension::FunctionDoc zt_norm;

PyObject* PyBobLearnEM_tNorm(PyObject*, PyObject* args, PyObject* kwargs);
extern bob::extension::FunctionDoc t_norm;

PyObject* PyBobLearnEM_zNorm(PyObject*, PyObject* args, PyObject* kwargs);
extern bob::extension::FunctionDoc z_norm;


//Linear scoring
PyObject* PyBobLearnEM_linear_scoring(PyObject*, PyObject* args, PyObject* kwargs);
extern bob::extension::FunctionDoc linear_scoring1;
extern bob::extension::FunctionDoc linear_scoring2;
extern bob::extension::FunctionDoc linear_scoring3;

#endif // BOB_LEARN_EM_MAIN_H

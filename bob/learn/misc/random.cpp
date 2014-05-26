/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 * @date Mon Jul 11 18:31:22 2011 +0200
 *
 * @brief Bindings for random number generation.
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "ndarray.h"
#include <boost/make_shared.hpp>
#include <boost/random.hpp>

using namespace boost::python;

template <typename T>
static boost::shared_ptr<boost::mt19937> make_with_seed(T s) {
  return boost::make_shared<boost::mt19937>(s);
}

template <typename T>
static void set_seed(boost::mt19937& o, T s) {
  o.seed(s);
}

void bind_core_random () {
  class_<boost::mt19937, boost::shared_ptr<boost::mt19937> >("mt19937",
      "A Mersenne-Twister Random Number Generator (RNG)\n" \
      "\n" \
      "A Random Number Generator (RNG) based on the work 'Mersenne Twister: A 623-dimensionally equidistributed uniform pseudo-random number generator, Makoto Matsumoto and Takuji Nishimura, ACM Transactions on Modeling and Computer Simulation: Special Issue on Uniform Random Number Generation, Vol. 8, No. 1, January 1998, pp. 3-30'", init<>((arg("self")), "Default constructor"))
    .def("__init__", make_constructor(&make_with_seed<int64_t>, default_call_policies(), (arg("seed"))), "Builds a new generator with a specific seed")
    .def("__init__", make_constructor(&make_with_seed<double>, default_call_policies(), (arg("seed"))), "Builds a new generator with a specific seed")
    .def("seed", &set_seed<double>, (arg("self"), arg("seed")), "Sets my internal seed using a floating-point number")
    .def("seed", &set_seed<int64_t>, (arg("self"), arg("seed")), "Sets my internal seed using an integer")
    .def(self == self)
    .def(self != self)
    ;
}

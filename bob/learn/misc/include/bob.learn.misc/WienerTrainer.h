/**
 * @date Fri Sep 30 16:58:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Trainer for the Wiener Machine
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_LEARN_MISC_WIENER_TRAINER_H
#define BOB_LEARN_MISC_WIENER_TRAINER_H

#include <bob.learn.misc/Trainer.h>
#include <bob.learn.misc/WienerMachine.h>
#include <blitz/array.h>

namespace bob { namespace learn { namespace misc {

/**
 * @brief Sets a Wiener machine to perform a Wiener filtering, using the
 * Fourier statistics of a given dataset.\n
 *
 * Reference:\n
 * "Computer Vision: Algorithms and Applications", Richard Szeliski
 * (Part 3.4.3)
 */
class WienerTrainer: Trainer<bob::learn::misc::WienerMachine, blitz::Array<double,3> >
{
  public: //api
    /**
     * @brief Default constructor.
     * Initializes a new Wiener trainer.
     */
    WienerTrainer();

    /**
     * @brief Copy constructor
     */
    WienerTrainer(const WienerTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~WienerTrainer();

    /**
     * @brief Assignment operator
     */
    WienerTrainer& operator=(const WienerTrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const WienerTrainer& other) const;
    /**
     * @brief Not equal to
     */
    bool operator!=(const WienerTrainer& other) const;
   /**
     * @brief Similar to
     */
    bool is_similar_to(const WienerTrainer& other, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Trains the WienerMachine to perform the filtering.
     */
    virtual void train(bob::learn::misc::WienerMachine& machine,
        const blitz::Array<double,3>& data);

  private: //representation
};

} } } // namespaces

#endif /* BOB_LEARN_MISC_WIENER_TRAINER_H */

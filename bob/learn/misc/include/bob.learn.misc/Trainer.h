/**
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */
#ifndef BOB_LEARN_MISC_TRAINER_H
#define BOB_LEARN_MISC_TRAINER_H

/**
 * @addtogroup TRAINER trainer
 * @brief Trainer module API
 */
namespace bob { namespace learn { namespace misc {

/**
 * @brief Root class for all trainers
 */
template<class T_machine, class T_sampler>
class Trainer
{
public:
  virtual ~Trainer() {};

  /**
   * @brief Train a \c machine using a sampler
   *
   * @param machine machine to train
   * @param sampler sampler that provides training data
   */
  virtual void train(T_machine& machine, const T_sampler& sampler) = 0;
};

} } } // namespaces

#endif // BOB_LEARN_MISC_TRAINER_H

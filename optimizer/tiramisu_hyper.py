#! /usr/bin/env python
#
# Stephen Vondenstein, Matthew Buckley
# 10/11/2018
#

import os
import tensorflow as tf
from models.tiramisu import Tiramisu as tiramisu
import hyperengine as hype

class HyperEngineOptimizer:
  def __init__(self, sess, model, data, config):
    self.config = config

    # Initialize local variables
    self.model = model
    self.config = config
    self.sess = sess
    self.data_loader = data

    # Initialize all variables of the graph
    self.init = tf.global_variables_initializer(), tf.local_variables_initializer()
    self.sess.run(self.init)

    # Load the model
    self.model.load(self.sess)

    # _, _, self.training, self.image_name = tf.get_collection('inputs')
    # self.image = tf.get_collection('out')

  def solver_generator(self, params):
    solver_params = {
      'batch_size': 16,
      'eval_batch_size': 16,
      'epochs': 32,
      'evaluate_test': True,
      'eval_flexible': False,
      'save_dir':
      # 'save_accuracy_limit': 0.9930,
    }
    tiramisu(config=params)
    solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
    return solver




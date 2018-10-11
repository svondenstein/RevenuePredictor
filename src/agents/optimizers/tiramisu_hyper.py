#! /usr/bin/env python
#
# Stephen Vondenstein, Matthew Buckley
# 10/11/2018
#

from src.agents.baseagent import BaseAgent
from src.models.tiramisu import Tiramisu as tiramisu
import hyperengine as hype


class HyperEngineOptimizer(BaseAgent):
  def __init__(self, config):
    BaseAgent.__init__(self, config)

    # Load the model
    self.model.load(self.sess)

    # _, _, self.training, self.image_name = tf.get_collection('inputs')
    # self.image = tf.get_collection('out')

  def solver_generator(self, params):
    solver_params = {
      'batch_size': 16,
      'eval_batch_size': 16,
      'epochs': self.config.e,
      'evaluate_test': True,
      'eval_flexible': False,
      # 'save_dir':
      # 'save_accuracy_limit': 0.9930,
    }
    tiramisu(config=params)
    solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
    return solver




#! /usr/bin/env python
#
# Stephen Vondenstein, Matthew Buckley
# 10/11/2018
#

from src.agents.baseagent import BaseAgent
from src.models.tiramisu import Tiramisu
import hyperengine as hype


class HyperEngineOptimizer(BaseAgent):
  def __init__(self, config):
    BaseAgent.__init__(self, config)



  def solver_generator(self, params):
    solver_params = {
      'batch_size': self.config.
      'eval_batch_size': 16,
      'epochs': self.config.e,
      'evaluate_test': True,
      'eval_flexible': False,
      # 'save_dir':
      # 'save_accuracy_limit': 0.9930,
    }
    Tiramisu(config=params)
    solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
    return solver

  hyper_params_spec = {
    'optimizer': {
      'learning_rate': 10**hype.spec.uniform(-5, -1),          # makes the continuous range [0.1, 0.001]
      'epsilon': 1e-8,                                    # constants work too
    },
    'conv': {
      'filters': [[3, 3, hype.spec.choice(range(32, 48))],     # an integer between [32, 48]
                  [3, 3, hype.spec.choice(range(64, 96))],     # an integer between [64, 96]
                  [3, 3, hype.spec.choice(range(128, 192))]],  # an integer between [128, 192]
      'activation': hype.spec.choice(['relu','prelu','elu']),  # a categorical range: 1 of 3 activations
      'down_sample': {
        'size': [2, 2],
        'pooling': hype.spec.choice(['max_pool', 'avg_pool'])  # a categorical range: 1 of 2 pooling methods
      },
      'residual': hype.spec.random_bool(),                     # either True or False
      'dropout': hype.spec.uniform(0.75, 1.0),                 # a uniform continuous range
    },
  }
  strategy_params = {
    'io_load_dir': 'temp-cifar10/example-2-2',
    'io_save_dir': 'temp-cifar10/example-2-2',
  }




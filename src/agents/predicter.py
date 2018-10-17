#
# Stephen Vondenstein, Matthew Buckley
# 10/10/2018
#
from tqdm import tqdm
import tensorflow as tf
from src.agents.baseagent import BaseAgent

from src.helpers.postprocess import process

class Predicter(BaseAgent):
    def __init__(self, model, config):
        BaseAgent.__init__(self, config)

        # Initialize local variables
        self.model = model

        # Load the model
        self.image, _, self.training = tf.get_collection('inputs')
        self.image = tf.get_collection('out')

    def predict(self):
    # Initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_infer), total=self.data_loader.num_iterations_infer, desc="Predicting ")
        with tf.Session() as sess:
            self.load(sess)

            # Initialize all variables of the graph
            self.init = tf.global_variables_initializer(), tf.local_variables_initializer()
            sess.run(self.init)

            # Initialize dataset
            self.data_loader.initialize(sess, 'infer')
            image, _, name, _ = sess.run(self.data_loader.get_data())

            # Iterate over batches
            for cur_it in tt:
                output_image = sess.run([self.image], feed_dict={self.training: False,
                                                          self.image: image})
                process(output_image, name, self.config)

        tt.close()

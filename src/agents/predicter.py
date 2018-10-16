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
        _, _, self.training, self.image_name = tf.get_collection('inputs')
        self.image = tf.get_collection('out')

    def predict(self):
    # Initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_infer), total=self.data_loader.num_iterations_infer, desc="Predicting ")
        with tf.Session() as sess:
            self.load(sess)

            # Initialize all variables of the graph
            self.init = tf.global_variables_initializer(), tf.local_variables_initializer()
            self.run(self.init)

            # Initialize dataset
            self.data_loader.initialize(sess, 'infer')
            
            # Iterate over batches
            for cur_it in tt:
                image, image_name = sess.run([self.image, self.image_name], feed_dict={self.training: False})
                process(image, image_name, self.config)

        tt.close()
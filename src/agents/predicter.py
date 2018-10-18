#
# Stephen Vondenstein, Matthew Buckley
# 10/10/2018
#
from tqdm import tqdm
import tensorflow as tf
from src.agents.baseagent import BaseAgent
from src.helpers.postprocess import process

class Predicter(BaseAgent):
    def __init__(self, config):
        super(Predicter, self).__init__(config)

        # Initialize variables
        self.image, _, self.training, self.depth = tf.get_collection('inputs')
        self.out = tf.get_collection('out')

    def predict(self):

        with tf.Session() as sess:
            self.load(sess)

            # Initialize tqdm
            tt = tqdm(range(self.data_loader.num_iterations_infer), total=self.data_loader.num_iterations_infer, desc="Predicting ")

            # Initialize all variables of the graph
            self.init = tf.global_variables_initializer(), tf.local_variables_initializer()
            sess.run(self.init)

            # Initialize dataset
            self.data_loader.initialize(sess, 'infer')
            next_item = self.data_loader.get_data()

            # Iterate over batches
            for cur_it in tt:
                image, _, name, depth = sess.run(next_item)
                output_image = sess.run(self.out,
                                            feed_dict={self.training: False,
                                                        self.image: image,
                                                        self.depth: depth})
                process(output_image, name, self.config)

        tt.close()

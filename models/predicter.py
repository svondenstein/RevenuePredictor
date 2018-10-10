#
# Stephen Vondenstein, Matthew Buckley
# 10/10/2018
#
from tqdm import tqdm
import tensorflow as tf

from helpers.postprocess import process

class Predicter:
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

        _, _, self.training, self.image_name = tf.get_collection('inputs')
        self.image = tf.get_collection('out')

    def predict(self):
        # Initialize dataset
        self.data_loader.initialize(self.sess, 'infer')

        # Initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_infer), total=self.data_loader.num_iterations_infer,
                  desc="Predicting ")

        self.model.load(self.sess)

        # Iterate over batches
        for cur_it in tt:
            image, image_name = self.sess.run([self.image, self.image_name], feed_dict={self.training: False})
            process(image, image_name, self.config)

        tt.close()

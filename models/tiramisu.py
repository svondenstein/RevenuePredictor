#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import tensorflow as tf

class Tiramisu:
    def __init__(self, data_loader, config):
        # Configuration parameters
        self.config = config

        # Step and epoch tensors to use as counters
        self.cur_epoch_tensor = None
        self.increment_cur_epoch_tensor = None
        self.global_step_tensor = None
        self.increment_global_step_tensor = None

        # Initialize step counter
        self.init_global_step()

        # Initialize epoch counter
        self.init_cur_epoch()

        # Initialize data loader
        self.data_loader = data_loader

        # Define local variables


        # Build model and initialize saver
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Build the tensorflow graph and define the loss
        pass

    # Save checkpoint
    def save(self, sess):
        print('Saving model to {}...\n').format(self.config.checkpoint_dir)
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved.")

    # Load checkpoint
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print('Loading model checkpoint {} ...\n').format(latest_checkpoint)
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded.")

    # Initialize epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # Initialize step counter
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    # Initialize tensorflow saver used for saving checkpoints
    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
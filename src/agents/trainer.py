#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
from tqdm import tqdm
import tensorflow as tf
from src.agents.baseagent import BaseAgent

from src.utils.utility import AverageMeter

class Trainer(BaseAgent):
    def __init__(self, model, config):
        BaseAgent.__init__(self, config)

        # Initialize local variables
        self.model = model

        # Step and epoch tensors to use as counters
        self.epoch_tensor = None
        self.increment_epoch_tensor = None

        # Initialize epoch counter
        self.init_epoch()

        # Initialize variables
        self.image, self.mask, self.training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node = tf.get_collection('train')

    # Training loop
    def train(self):
        with tf.Session() as sess:

            # Load the model
            self.load(sess)

            # Initialize all variables of the graph
            self.init = tf.global_variables_initializer(), tf.local_variables_initializer()
            sess.run(self.init)

            for cur_epoch in range(self.epoch_tensor.eval(sess), self.config.epochs + 1, 1):
                self.train_epoch(sess,cur_epoch)
                sess.run(self.increment_epoch_tensor)
                self.test(sess,cur_epoch)

    # Train one epoch
    def train_epoch(self, sess, epoch=None):
        # Initialize dataset
        self.data_loader.initialize(sess, 'train')
        next_item = self.data_loader.get_data()

        # Initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="Epoch {} ".format(epoch + 1))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            image, mask, _, _ = sess.run(self.data_loader.get_data())
            _, loss, acc = sess.run([self.train_op, self.loss_node, self.acc_node], feed_dict={self.training: True,
                                                                                               self.mask: mask,
                                                                                               self.image: image})
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        sess.run(self.increment_epoch_tensor)

        self.save(sess)

        print('Epoch {} loss:{:.4f} -- acc:{:.4f}'.format(epoch + 1, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()

    def test(self, sess, epoch):
        # Initialize dataset
        self.data_loader.initialize(sess, 'test')
        next_item = self.data_loader.get_val()

        # Initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Val {} ".format(epoch + 1))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            image, mask, _, _ = sess.run(next_item)
            loss, acc = sess.run([self.loss_node, self.acc_node], feed_dict={self.training: False,
                                                                             self.mask: mask,
                                                                             self.image: image})
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        print('Val {} loss:{:.4f} -- acc:{:.4f}'.format(epoch + 1, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()

    # Initialize epoch counter
    def init_epoch(self):
        with tf.variable_scope('epoch'):
            self.epoch_tensor = tf.Variable(0, trainable=False, name='epoch')
            self.increment_epoch_tensor = self.epoch_tensor.assign(self.epoch_tensor + 1)

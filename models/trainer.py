#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
from tqdm import tqdm
import tensorflow as tf

from utils.utility import AverageMeter


class Trainer:
    def __init__(self, sess, model, data, config, logger):
        self.config = config

        # Initialize local variables
        self.model = model
        self.config = config
        self.sess = sess
        self.data_loader = data
        self.logger = logger

        # Initialize all variables of the graph
        self.init = tf.global_variables_initializer(), tf.local_variables_initializer()
        self.sess.run(self.init)

        # Load the model
        self.model.load(self.sess)

        self.image, self.mask, self.training, _ = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node = tf.get_collection('train')

    # Training loop
    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)

    # Train one epoch
    def train_epoch(self, epoch=None):
        # Initialize dataset
        self.data_loader.initialize(self.sess, 'train')

        # Initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="Epoch {} ".format(epoch + 1))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            loss, acc = self.train_step()
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        self.sess.run(self.model.increment_global_epoch_tensor)

        # summarize
        summaries_dict = {'train/loss_per_epoch': loss_per_epoch.val,
                          'train/acc_per_epoch': acc_per_epoch.val}
        self.logger.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        self.model.save(self.sess)

        print('Epoch {} loss:{:.4f} -- acc:{:.4f}'.format(epoch + 1, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()


    def train_step(self):
        _, loss, acc = self.sess.run([self.train_op, self.loss_node, self.acc_node], feed_dict={self.training: True})
        return loss, acc

    def test(self, epoch):
        # Initialize dataset
        self.data_loader.initialize(self.sess, 'test')

        # Initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Val {} ".format(epoch + 1))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            loss, acc = self.sess.run([self.loss_node, self.acc_node], feed_dict={self.training: False})
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        # summarize
        summaries_dict = {'test/loss_per_epoch': loss_per_epoch.val,
                          'test/acc_per_epoch': acc_per_epoch.val}
        self.logger.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print('Val {} loss:{:.4f} -- acc:{:.4f}'.format(epoch + 1, loss_per_epoch.val, acc_per_epoch.val))

        tt.close()
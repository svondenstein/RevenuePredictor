#
# Stephen Vondenstein, Matthew Buckley
# 10/13/2018
#
import tensorflow as tf
import os


class Logger:
    def __init__(self, sess, summary_dir, scalar_tags=None, images_tags=None):
        self.sess = sess

        self.scalar_tags = scalar_tags
        self.images_tags = images_tags

        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}

        self.init_summary_ops()

        self.summary_writer = tf.summary.FileWriter(summary_dir)

    def set_summaries(self, scalar_tags=None, images_tags=None):
        self.scalar_tags = scalar_tags
        self.images_tags = images_tags
        self.init_summary_ops()

    def init_summary_ops(self):
        with tf.variable_scope('summary_ops'):
            if self.scalar_tags is not None:
                for tag in self.scalar_tags:
                    self.summary_tags += [tag]
                    self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                    self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            if self.images_tags is not None:
                for tag, shape in self.images_tags:
                    self.summary_tags += [tag]
                    self.summary_placeholders[tag] = tf.placeholder('float32', shape, name=tag)
                    self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=10)

    def summarize(self, step, summaries_dict=None, summaries_merged=None):
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)

            if hasattr(self, 'experiment') and self.experiment is not None:
                self.experiment.log_multiple_metrics(summaries_dict, step=step)

    def finalize(self):
        self.summary_writer.flush()

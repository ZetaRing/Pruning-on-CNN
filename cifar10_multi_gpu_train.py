# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using multiple GPU's with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import pdb
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10
import cifar10_eval

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/wind/Desktop/cifarReportWeek1/cifar10_exp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 30000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('keep_prob', 0.5,
                            """With probability to keep connections in dropout layer.""")
#tf.app.flags.DEFINE_boolean('is_training', True,
#                            """Whether training or testing.""")
tf.app.flags.DEFINE_boolean('reload', True,
                            """Whether to reload or train from scratch.""")
tf.app.flags.DEFINE_boolean('soft', True,
                            """Choose deterministic or stochastic pruning.""")
tf.app.flags.DEFINE_float('precision_fact',499,
                            """How important precision rate is when pruning""")
tf.app.flags.DEFINE_float('compress_fact',1,
                            """How important compress rate is when pruning""")

def tower_loss(scope):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  images, labels = cifar10.distorted_inputs()

  # Build inference Graph.
  logits = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  # print("trainable---------------------")
  # for i in tf.trainable_variables():
  #   print(i)

  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #print("tuple-----------------------")
    #print(grad_and_vars)
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    print("v--------------------------")
    print(v)
    with tf.variable_scope('mask') as scope:
      mask = tf.get_variable(v.op.name,
              shape=v.get_shape(),
              initializer=tf.constant_initializer(1.0, tf.int8),
              trainable=False)
    grad = tf.multiply(grad, mask)
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def pruning(k, scale):
  """Define pruning ops over all weights.

  Add summary for "Loss" and "Loss/avg".
  Args:
    k: The slope of the probabilistic curve.
    scale: Scale factor multiplied by weights

  Returns:
    Pruning ops, would prune the weights when run.
  """
  pruning_ops = []
  for x in tf.trainable_variables():
    if ('conv' in x.op.name or 'local' in x.op.name or 'softmax_linear' in x.op.name) and 'weights' in x.op.name:
      zeros = tf.zeros_like(x)
      ones = tf.ones_like(x)
      sigma = tf.reduce_max(tf.abs(x))*scale

      #Add a mask to constrain the weights being pruned to be zeros.
      with tf.variable_scope('mask', reuse=True) as scope:
        mask = tf.get_variable(x.op.name)

      # Prune the weights with probabilistic or deterministic method.
      if FLAGS.soft:
        uni = tf.random_uniform(x.get_shape(), minval=0, maxval=1)
        # p_x = tf.where(tf.greater(tf.abs(x), sigma), ones, tf.abs(x)/sigma)
        k_x = tf.where(tf.greater(tf.abs(x)*k, 1.0), ones, tf.abs(x)*k)
        p_x = tf.where(tf.greater(tf.abs(x), sigma), ones, k_x)

        prun_op = x.assign(tf.where(tf.greater(p_x, uni), x, zeros))
        constrain_op = mask.assign(tf.where(tf.greater(p_x, uni), ones, zeros))

      else:
        prun_op = x.assign(tf.where(tf.greater(tf.abs(x), sigma), x, zeros))
        constrain_op = mask.assign(tf.where(tf.greater(tf.abs(x), sigma), ones, zeros))
        
      pruning_ops.append(prun_op)
      pruning_ops.append(constrain_op)
  return tf.group(*pruning_ops)


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=False)
    # opt = tf.train.AdamOptimizer()

    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    # pdb.set_trace()
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Prune the network iterately.s
    #pruning_ops = pruning(k = 1.25, scale=0.02)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to 
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    if FLAGS.reload:
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      else:
        step = 1
    else:
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
      tf.gfile.MakeDirs(FLAGS.train_dir)
      step = 1

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    while step <= FLAGS.max_steps:
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
      
      #pruning
      if step % 3000 == 0:
        # Init precision and compress
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        # Evalute precision and compress rate
        initPrecision,initCompress = cifar10_eval.evaluate()
        print("initPrecision: %.4f , initCompress: %.4f" % \
             (initPrecision,initCompress))

        scaleFounder = float(0.01)
        while True:
          # Prune the network iterately.s
          pruning_ops = pruning(k = 1.5, scale=scaleFounder)
          sess.run(pruning_ops)
          # Save the model for eval
          # tip: no need for packup because pruning scale is  monotonic increasing
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
          # Evalute precision and compress rate
          precision,compress = cifar10_eval.evaluate()
          print("precision: %.4f , compress: %.4f" % (precision,compress))

          scaleFounder = scaleFounder + 0.01
          
          if(
              (FLAGS.precision_fact * (initPrecision - precision) - 
               FLAGS.compress_fact * (initCompress - compress) > 0 )or \
              scaleFounder == 1.0
            ):break

      # Save the model checkpoint periodically.
      if step % 500 == 0 or step == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        #cifar10_eval.evaluate()
      step += 1

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

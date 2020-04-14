# coding:utf-8

import re
import sys
import logging
import argparse
import collections
import tensorflow as tf
from pathlib import Path

import config as cg
import model_LEGO as lego
import function_toolkit as ft
from model import BertEncoder
from data_utils_2 import train_input_fn, server_input_fn

from log import log_info as _info
from log import log_error as _error

MAIN_PATH = Path(__file__).absolute().parent

# log record
class Setup(object):
    """Setup logging"""
    def __init__(self, log_name='tensorflow', path=str(MAIN_PATH / 'log')):
        Path('log').mkdir(exist_ok=True)
        tf.compat.v1.logging.set_verbosity(logging.INFO)
        handlers = [logging.FileHandler(str(MAIN_PATH / 'log/main.log')),
                    logging.StreamHandler(sys.stdout)]
        logging.getLogger('tensorflow').handlers = handlers
setup = Setup()

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)

def gather_indexs(sequence_output, sentiment_mask_indices):
  shape = ft.get_shape_list(sequence_output, expected_rank=3)
  batch_size = shape[0]
  seq_length = shape[1]
  width = shape[2]

  # [b, 1]
  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  # [b, x]
  flat_positions = tf.reshape(flat_offsets + sentiment_mask_indices, [-1])
  # [b * s, w]
  flat_sequence_output = tf.reshape(sequence_output, [batch_size * seq_length, width])
  # [b * x, w]
  output_tensor = tf.gather(flat_sequence_output, flat_positions)

  return output_tensor

def get_true_sequence(sentiment_labels):
  # [0.7, 0.5, -0.2, -2, -2] -> [2.7, 2.5, 1.8, 0, 0]
  # -> [True, True, True, False, False] -> [1.0, 1.0, 1.0, 0.0, 0.0]
  sentiment_labels = tf.reshape(sentiment_labels, [-1]) + 1
  sentiment_labels = tf.cast(sentiment_labels, dtype=tf.bool)
  true_sequence = tf.cast(sentiment_labels, dtype=tf.float32)

  return true_sequence

def calculate_mse_loss(model_output, true_label, true_sequence):
  """This is used for calculating the mse loss.
  
  Args:
    model_output: (batch_size * seq_length, mask_padding_size).
    true_label: (batch, seq_length, mask_padding_size).
    true_sequence: (batch * seq_length * mask_padding_size).

  Returns:
    mse_loss: tf.float32.
  """
  batch_size = tf.cast(ft.get_shape_list(model_output, expected_rank=2)[0], dtype=tf.float32)
  # flatten the tensor
  model_output_flatten = tf.reshape(model_output, [-1])
  true_label_flatten = tf.reshape(true_label, [-1])
  # get actual length without mask, cause the following mse calculation ignore the mask
  length = tf.reduce_sum(true_sequence)

  mse_loss = tf.reduce_sum(
    tf.pow((model_output_flatten - true_label_flatten), 2) * true_sequence) / batch_size

  return mse_loss

def model_fn_builder(init_checkpoint=None):
  """returns `model_fn` closure for the Estimator."""

  def model_fn(features, labels, mode, params):
    # features name and shape
    for name in sorted(features.keys()):
      tf.logging.info(' name = {}, shape = {}'.format(name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # get data
    input_data = features['input_data']
    input_mask = features['input_mask']
    if mode == tf.estimator.ModeKeys.TRAIN:
      sentiment_labels = features['sentiment_labels']
      sentiment_mask_indices = features['sentiment_mask_indices']
      true_length_from_data = features['true_length']

    # build model
    model = BertEncoder(
      config=cg.BertEncoderConfig,
      is_training=is_training,
      input_ids=input_data,
      input_mask=input_mask)
    
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    # [cls] output -> [b, h]
    cls_output = model.get_cls_output()
    # sequence_output -> [b, s, h], do not contain [CLS], because the mask indices do not shift
    sequence_output = model.get_sequence_output()[:, 1:, :]

    # project the hidden size to the num_classes
    with tf.variable_scope('final_output'):
      # [b, num_classes]
      output_logits = tf.layers.dense(
        cls_output,
        cg.BertEncoderConfig.num_classes,
        name='final_output',
        kernel_initializer=ft.create_initializer(initializer_range=cg.BertEncoderConfig.initializer_range))

    if mode == tf.estimator.ModeKeys.PREDICT:
      output_softmax = tf.nn.softmax(output_logits, axis=-1)
      output_result = tf.argmax(output_softmax, axis=-1)
      predictions = {'predict': output_result}
      output_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
      if mode == tf.estimator.ModeKeys.TRAIN:
        # masked_output -> [b * x, h]
        masked_output = gather_indexs(sequence_output, sentiment_mask_indices)

        # get output for word polarity prediction
        with tf.variable_scope('sentiment_project'):
          # [b * x, 2]
          output_sentiment = tf.layers.dense(
            masked_output,
            2,
            name='final_output',
            kernel_initializer=ft.create_initializer(initializer_range=cg.BertEncoderConfig.initializer_range))
        # output_sentiment_probs = tf.nn.softmax(output_sentiment, axis=-1)

        batch_size = tf.cast(ft.get_shape_list(labels, expected_rank=1)[0], dtype=tf.float32)
        # cross-entropy loss
        cls_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels,
          logits=output_logits)) / batch_size

        # mse loss
        # # Regression Model
        true_sequence = get_true_sequence(true_length_from_data)
        # mse_loss = calculate_mse_loss(
        #   output_sentiment, sentiment_labels, true_sequence)

        # # Classification Model
        true_label_flatten = tf.reshape(sentiment_labels, [-1])
        mse_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=true_label_flatten,
          logits=output_sentiment) * true_sequence) / tf.reduce_sum(true_sequence)

        loss = cls_loss + mse_loss
        # loss = cls_loss

        learning_rate = tf.train.polynomial_decay(cg.learning_rate,
                                  tf.train.get_or_create_global_step(),
                                  cg.train_steps,
                                  end_learning_rate=cg.lr_limit,
                                  power=1.0,
                                  cycle=False)

        lr = tf.maximum(tf.constant(cg.lr_limit), learning_rate)
        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=cg.colocate_gradients_with_ops)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=tf.train.get_global_step())

        current_steps = tf.train.get_or_create_global_step()
        logging_hook = tf.train.LoggingTensorHook(
          {'step' : current_steps, 'loss' : loss, 'cls_loss' : cls_loss, 'mse_loss' : mse_loss, 'lr' : lr}, 
          every_n_iter=cg.print_info_interval)

        output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
      elif mode == tf.estimator.ModeKeys.EVAL:
        # TODO
        raise NotImplementedError
    
    return output_spec
  
  return model_fn
      
def main():
  Path(cg.save_model_path).mkdir(exist_ok=True)

  model_fn = model_fn_builder()

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True

  run_config = tf.contrib.tpu.RunConfig(
    session_config=gpu_config,
    keep_checkpoint_max=cg.keep_checkpoint_max,
    save_checkpoints_steps=cg.save_checkpoints_steps,
    model_dir=cg.save_model_path)

  estimator = tf.estimator.Estimator(model_fn, config=run_config)
  estimator.train(train_input_fn)

def package_model(ckpt_path, pb_path):
  model_fn = model_fn_builder()
  estimator = tf.estimator.Estimator(model_fn, ckpt_path)
  estimator.export_saved_model(pb_path, server_input_fn)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=_info('python train.py [train | package]', head='USAGE:'))
  parser.add_argument('mode')
  
  args = parser.parse_args()
  mode = args.mode
  if mode == 'train':
    main()
  elif mode == 'package':
    package_model(cg.save_model_path, cg.pb_model_path)
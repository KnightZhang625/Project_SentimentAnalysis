# coding:utf-8

import codecs
import pickle
import functools
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent

import config as cg
from data_utils import provide_batch_idx, process_line, padding_data, make_mask

from log import log_info as _info
from log import log_error as _error

def restore_model(pb_path):
  """Restore the latest model from the given path."""
  subdirs = [x for x in Path(pb_path).iterdir()
             if x.is_dir() and 'temp' not in str(x)]
  latest_model = str(sorted(subdirs)[-1])
  predict_fn = predictor.from_saved_model(latest_model)

  return predict_fn

def predict(model, test_data_path, batch_size=32):
  with codecs.open(test_data_path, 'rb') as file:
    data = pickle.load(file)
  _info('The total test data length: {}.'.format(len(data)))

  predict_result_set = []
  for (start, end) in provide_batch_idx(len(data), batch_size):
    data_batch = data[start:end]
    sentences = [data[1] for data in data_batch]
    # labels = [data[0] for data in data_batch]

    sentences_idx = list(map(process_line, sentences))
    sentences_idx_padded = padding_data(sentences_idx)
    input_mask = list(map(make_mask, sentences_idx_padded))

    features = {'input_data': sentences_idx_padded,
                'input_mask': input_mask}
    
    predictions = model(features)
    predict_results = predictions['predict']

    predict_result_set.extend(predict_results)
  
  return predict_result_set

if __name__ == '__main__':
  model = restore_model(cg.pb_model_path)
  
  predict_pos = predict(model, MAIN_PATH / 'data/Stanford_Data_binary/test_train_pos.bin')
  pos_accuracy = sum(predict_pos) / len(predict_pos)
  predict_neg = predict(model, MAIN_PATH / 'data/Stanford_Data_binary/test_train_neg.bin')
  neg_accuracy = 1 - sum(predict_neg) / len(predict_neg)

  _info('Predict positve accuracy: {}.'.format(pos_accuracy))
  _info('Predict negative accuracy: {}.'.format(neg_accuracy))
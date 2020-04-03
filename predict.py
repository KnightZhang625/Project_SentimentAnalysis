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
  
  predict_result_set = []
  for (start, end) in provide_batch_idx(len(data), batch_size):
    data_batch = data[start:end]
    sentences = [data[1] for data in data_batch]
    labels = [data[0] for data in data_batch]

    sentences_idx = list(map(process_line, sentences))
    sentences_idx_padded = padding_data(sentences_idx)
    input_mask = list(map(make_mask, sentences_idx_padded))

    features = {'input_data': sentences_idx_padded,
                'input_mask': input_mask}
    
    predictions = model(features)
    predict_results = predictions['predict']

    return predict_results

if __name__ == '__main__':
  model = restore_model(cg.pb_model_path)
  predict(model, MAIN_PATH / 'data/Stanford_Data_binary/test_pos.bin')
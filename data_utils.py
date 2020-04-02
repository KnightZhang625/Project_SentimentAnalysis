# coding:utf-8

import codecs
import pickle
import random
import functools
import tensorflow as tf
# tf.enable_eager_execution()

from log import log_info as _info
from log import log_error as _error
from log import print_process as _process

from config import batch_size
from config import TRAIN_POS_DATA_PATH as pos_data_path
from config import TRAIN_NEG_DATA_PATH as neg_data_path
from config import VOCAB_IDX_PATH as vocab_idx_path
from config import INT_PATH as int_path
from config import NEG_PATH as neg_path

def load_dict():
  """load the necessary dictionaries."""
  global vocab_idx
  global int_vocab
  global neg_vocab
  with codecs.open(vocab_idx_path, 'rb') as file_vocab, \
       codecs.open(int_path, 'rb') as file_int, \
       codecs.open(neg_path, 'rb') as file_neg:
      vocab_idx = pickle.load(file_vocab)
      int_vocab = pickle.load(file_int)
      neg_vocab = pickle.load(file_neg)
load_dict()

def provide_batch_idx(data_length, batch_size):
  """provide the start and end indices for each batch."""
  divided_or_not = True if data_length % batch_size == 0 else False
  total_batch_number = (data_length // batch_size) if divided_or_not \
    else (data_length // batch_size + 1)
  
  # it's okay if the end slice is larger than the last batch ending index
  for num in range(total_batch_number):
    yield (num * batch_size, num * batch_size + batch_size)

padding_func = lambda line, max_length: line + [vocab_idx['<padding>'] for _ in range(max_length - len(line))]
def padding_data(data_batch):
  """padding each batch to the same length."""
  max_length = max([len(data) for data in data_batch])
  padding_func_with_args = functools.partial(padding_func, max_length=max_length)
  data_batch_padded = list(map(padding_func_with_args, data_batch))
  return data_batch_padded

"""should, could"""
"""<br /><br />, ., ,"""
def process_line(line):
  """Tool for processing each line.
  
    Steps:
      1. split;
      2. extract keywords, keep the origal position;      
  """
  # split the sentence
  line_set = [line]
  for split_tag in ['<br /><br />', '.', ',']:
    cache = []
    for line in line_set:
      line = line.split(split_tag)
      cache.extend(line)
    line_set = cache
  
  # extract keywords
  final_line = []
  for idx, line in enumerate(line_set):
    line = line.split(' ')
    cache = []
    for v in line:
      if v in ['should', 'could']:
        break
      elif v in vocab_idx:
        cache.append(vocab_idx[v])
      elif v in int_vocab:
        score = int_vocab[v]
        if -3 <= score < -2:
          cache.append(vocab_idx['<int_-3>'])
        elif -2 <= score < -1:
          cache.append(vocab_idx['<int_-2>'])
        elif -1 <= score < 0:
          cache.append(vocab_idx['<int_-1>'])
        else:
          cache.append(vocab_idx['<int_0>'])
      elif v in neg_vocab:
        cache.append(vocab_idx['<negation>'])
      else:
        # ignore other words
        pass
    if idx < len(line_set) - 1 and len(cache) != 0:
      cache.append(vocab_idx['<seq>'])
      final_line.extend(cache)
  final_line.insert(0, vocab_idx['<cls>'])
  
  return final_line

intensive_negation_idx_set = [vocab_idx['<int_-3>'], vocab_idx['<int_-2>'], vocab_idx['<int_-1>'], vocab_idx['<int_0>'], vocab_idx['<negation>']]
not_word_idx_set = [vocab_idx['<padding>'], vocab_idx['<seq>'], vocab_idx['<cls>']]
def make_mask(data):
  mask = []
  for i_o, v_o in enumerate(data):
    if v_o == vocab_idx['<cls>']:
      cache = []
      for v in data:
        if v != vocab_idx['<padding>']:
          cache.append(1)
        else:
          cache.append(0)
    elif v_o == vocab_idx['<padding>']:   # padding
      cache = [0 for _ in range(len(data))]
    else:
      if v_o in intensive_negation_idx_set:
        cache = [0 for _ in range(len(data))]
        cache[i_o] = 1
        if data[i_o + 1] not in not_word_idx_set:
            cache[i_o + 1] = 1
      else:
        cache = []
        for i, v in enumerate(data):
          if v in intensive_negation_idx_set:
            if i_o - i == 1:
              cache.append(1)
            else:
              cache.append(0)
          else:
            if v != vocab_idx['<padding>']:
              cache.append(1)
            else:
              cache.append(0)
    mask.append(cache)
  
  return mask

def train_generator():
  """make train, test data."""
  # load the data
  with codecs.open(pos_data_path, 'rb') as file_p,\
       codecs.open(neg_data_path, 'rb') as file_n:
      pos_data = pickle.load(file_p)
      neg_data = pickle.load(file_n)
  assert len(pos_data) == len(neg_data), _error('Data distribution uneven.', head='ERROR')

  # shuffle the data
  train_data = pos_data + neg_data
  random.shuffle(train_data)
  
  # create batch
  for (start, end) in provide_batch_idx(len(train_data), batch_size):
    data_batch = train_data[start: end]
    sentences = [data[1] for data in data_batch]
    labels = [data[0] for data in data_batch]

    sentences_idx = list(map(process_line, sentences))
    sentences_idx_padded = padding_data(sentences_idx)
    yield [len(sen) for sen in sentences_idx_padded]

def train_input_fn():
  output_types = {'input_data': tf.int32, 
                  'input_mask': tf.int32}
  output_shapes = {'input_data': [None, None],
                   'input_mask': [None, None, None]}


if __name__ == '__main__':
  # for l in train_generator():
  #   print(l)
  #   input()

  # test_sentence = 'this is a test, i don\' know, this is cool.howerve this is cool, cool cool cool.<br /><br />hahaha, this is haha, go back.'
  # process_line(test_sentence)

  test_data = [0, 100, 200, 3, 100, 3, 2, 100, 1, 1]
  mask = make_mask(test_data)
  print(mask)
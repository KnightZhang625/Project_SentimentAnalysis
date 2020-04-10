# coding:utf-8

import copy
import nltk
import codecs
import pickle
import random
import functools
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from nltk import sent_tokenize, word_tokenize

from preprocess.SentiWordNet import get_sentiment, penn_to_wn, ps

from log import log_info as _info
from log import log_error as _error
from log import print_process as _process

from config import batch_size
from config import train_steps
from config import random_mask_prob
from config import VOCAB_IDX_PATH as vocab_idx_path
from config import TRAIN_POS_DATA_PATH as pos_data_path
from config import TRAIN_NEG_DATA_PATH as neg_data_path

"""save dictionary as binary format."""
def make_dict(path):
  vocab_idx = {}
  idx_vocab = {}
  with codecs.open(path, 'r', 'utf-8') as file:
    for i, vocab in enumerate(file):
      vocab = vocab.strip()
      if len(vocab) > 0:
        vocab_idx[vocab] = i
        idx_vocab[i] = vocab

  with codecs.open('data/vocab_idx.bin', 'wb') as file, \
       codecs.open('data/idx_vocab.bin', 'wb') as file_2:
      pickle.dump(vocab_idx, file)
      pickle.dump(idx_vocab, file_2) 

"""load the dictionary file."""
def load_dict():
  global vocab_idx
  with codecs.open(vocab_idx_path, 'rb') as file:
    vocab_idx = pickle.load(file)
load_dict()

"""provide the start and end indices for each batch."""
def provide_batch_idx(data_length, batch_size):
  divided_or_not = True if data_length % batch_size == 0 else False
  total_batch_number = (data_length // batch_size) if divided_or_not \
    else (data_length // batch_size + 1)
  
  # it's okay if the end slice is larger than the last batch ending index
  for num in range(total_batch_number):
    yield (num * batch_size, num * batch_size + batch_size)

"""padding data."""
padding_func = lambda line, max_length, padding_tag : \
                line + [padding_tag for _ in range(max_length - len(line))]
def padding_data(data, padding_tag=vocab_idx['[PAD]']):
  max_length = max([len(item) for item in data])
  padding_func_with_args = functools.partial(padding_func, 
                                             max_length=max_length, 
                                             padding_tag=padding_tag)
  return list(map(padding_func_with_args, data))

"""make mask."""
def make_mask(data, sentiment_indices):
  input_mask = []
  for i, idx in enumerate(data):
    if i not in sentiment_indices and idx != vocab_idx['[PAD]']:
      input_mask.append(1)
    else:
      input_mask.append(0)
  return input_mask

"""ramdom mask sentiment words."""
def random_mask(data):
  # clean data
  data = data.replace('<br />', ' ')
  # split sentence
  sentences_set = sent_tokenize(data)

  data_final = []
  word_polarity_labels = []
  mask_indices = []
  preb_sentence_length = 0  # this is used for shiftting mask_indices
  for sentence in sentences_set:
    # tokenize and stemming
    sentence_stem = [ps.stem(v) for v in word_tokenize(sentence)]
    # pos tag
    sentence_tagged = nltk.pos_tag(sentence_stem)
    # get necessary sentiment for each word
    sentence_sentiment = [get_sentiment(v, p) for (v, p) in sentence_tagged]
    # get sentiment words indices
    sentiment_indices = [i for i, item in enumerate(sentence_sentiment) if len(item) > 0]
    # number of sentiments to mask
    if len(sentiment_indices) == 0:
      # believe that one comment contains as least one useful sentence
      continue
    else:
      number_to_mask_cand = int(len(sentiment_indices) * random_mask_prob)
      number_to_mask = number_to_mask_cand if number_to_mask_cand >= 1 else 1
      indices_to_mask = sorted(random.sample(sentiment_indices, number_to_mask))
      mask_indices.extend([idx + preb_sentence_length for idx in copy.deepcopy(indices_to_mask)])
      preb_sentence_length += len(sentence_stem)

    # convert str to idx
    data_temp = []
    for i, vocab in enumerate(sentence_stem):
      if i in indices_to_mask:
        word_polarity_labels.append(sentence_sentiment[i])  # add labels
        data_temp.append(vocab_idx['[MASK]'])
      else:
        data_temp.append(vocab_idx[vocab] if vocab in vocab_idx
                          else vocab_idx['[UNK]'])
    data_temp.append(vocab_idx['[SEP]'])
    data_final.extend(data_temp)

  data_final.insert(0, vocab_idx['[CLS]'])
  
  return (data_final, word_polarity_labels, mask_indices)

"""extract sentiment, language model, word polarity features."""
def extract_features(data):
  sentences = [item[1] for item in data]
  labels = np.array([item[0] for item in data], dtype=np.int32)

  # Random Mask
  sentiment_featurs = list(map(random_mask, sentences))
  input_idx = [item[0] for item in sentiment_featurs]
  sentiment_labels = [item[1] for item in sentiment_featurs]
  sentiment_mask_indices = [item[2] for item in sentiment_featurs]

  # Padding
  input_idx_padded = np.array(padding_data(input_idx), dtype=np.int32)
  sentiment_labels_padded = np.array(padding_data(sentiment_labels, [-1, -1, -1]), dtype=np.float32)
  sentiment_mask_indices_padded = np.array(padding_data(sentiment_mask_indices, -1), dtype=np.int32)

  # Make Mask
  input_mask = np.array(list(map(make_mask, input_idx_padded, sentiment_mask_indices)), dtype=np.int32)

  features = {'input_data': input_idx_padded,
               'input_mask': input_mask,
               'sentiment_labels': sentiment_labels_padded,
               'sentiment_mask_indices': sentiment_mask_indices_padded}

  return (features, labels)

def train_generator():
  """generator to yield data."""
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
    data_batch = train_data[start : end]
    yield extract_features(data_batch)

def train_input_fn():
  output_types = {'input_data': tf.int32,
                  'input_mask': tf.int32,
                  'sentiment_labels': tf.float32,
                  'sentiment_mask_indices': tf.int32}
  output_shapes = {'input_data': [None, None],
                   'input_mask': [None, None],
                   'sentiment_labels': [None, None, 3],
                   'sentiment_mask_indices': [None, None]}
  
  dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_types=(output_types, tf.int32),
    output_shapes=(output_shapes, [None]))

  dataset = dataset.repeat(train_steps)

  return dataset

if __name__ == '__main__':
  for data in train_input_fn():
    print(data)
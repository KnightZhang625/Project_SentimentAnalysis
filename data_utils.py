# coding:utf-8

import codecs
import pickle
import random
import functools
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

from nltk.stem import PorterStemmer 
ps = PorterStemmer()

from log import log_info as _info
from log import log_error as _error
from log import print_process as _process

from config import batch_size
from config import train_steps
from config import TRAIN_POS_DATA_PATH as pos_data_path
from config import TRAIN_NEG_DATA_PATH as neg_data_path
from config import VOCAB_IDX_PATH as vocab_idx_path
from config import KEYWORDS_PATH as keywords_path
from config import INT_PATH as int_path
from config import NEG_PATH as neg_path
from config import BIGRAM_PATH as bigram_path
from config import BIGRAM_INT_PATH as bigram_int_path

"""load the necessary dictionaries."""
def load_dict():
  global vocab_idx
  global int_vocab
  global neg_vocab
  global keywords_vocab
  global bigram
  global bigram_int
  with codecs.open(vocab_idx_path, 'rb') as file_vocab, \
       codecs.open(int_path, 'rb') as file_int, \
       codecs.open(neg_path, 'rb') as file_neg, \
       codecs.open(keywords_path, 'rb') as file_key, \
       codecs.open(bigram_path, 'rb') as file_bi, \
       codecs.open(bigram_int_path, 'rb') as file_bi_int:
      vocab_idx = pickle.load(file_vocab)
      int_vocab = pickle.load(file_int)
      neg_vocab = pickle.load(file_neg)
      keywords_vocab = pickle.load(file_key)
      bigram = pickle.load(file_bi)
      bigram_int = pickle.load(file_bi_int)
load_dict()

"""provide the start and end indices for each batch."""
def provide_batch_idx(data_length, batch_size):
  divided_or_not = True if data_length % batch_size == 0 else False
  total_batch_number = (data_length // batch_size) if divided_or_not \
    else (data_length // batch_size + 1)
  
  # it's okay if the end slice is larger than the last batch ending index
  for num in range(total_batch_number):
    yield (num * batch_size, num * batch_size + batch_size)

"""padding each batch to the same length."""
padding_func = lambda line, max_length: line + [vocab_idx['<padding>'] for _ in range(max_length - len(line))]
def padding_data(data_batch):
  max_length = max([len(data) for data in data_batch])
  padding_func_with_args = functools.partial(padding_func, max_length=max_length)
  data_batch_padded = list(map(padding_func_with_args, data_batch))
  return data_batch_padded

"""return section score for intensive vocabs."""
def intense_section(score):
  if -3 <= score < -2:
    return vocab_idx['<int_-3>']
  elif -2 <= score < -1:
    return vocab_idx['<int_-2>']
  elif -1 <= score < 0:
    return vocab_idx['<int_-1>']
  else:
    return vocab_idx['<int_0>']

"""extract keywords, intensive vocab for each line."""
def process_line(line, include_bi=True):
  """Tool for processing each line. Split_Tag: `<br /><br />`, `.`, `,` 
  
    Steps:
      1. split;
      2. extract keywords, keep the origal position;      
  """
  def check_uni(uni_cand):
    """extract unigram words."""
    if uni_cand in keywords_vocab:
      return vocab_idx[keywords_vocab[uni_cand]]
    elif uni_cand in int_vocab:
      score = int_vocab[uni_cand]
      return intense_section(score)
    elif uni_cand in neg_vocab:
      return vocab_idx['<negation>']
    else:
      return None

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
  for line in line_set:
    # split the words
    line = line.strip().split(' ')
    # words indices cache for each line
    cache = []

    # extract bigram keywords
    if include_bi:
      n = 0   # indicate current index
      while n < len(line):
        next_n = n + 1  # index for the next word right after current n
        if next_n < len(line):
          bi_cand = ' '.join(line[n:next_n+1])  # candidate bigram
          if bi_cand in bigram:
            cache.append(vocab_idx[bigram[bi_cand]])
            n = next_n + 1
          elif bi_cand in bigram_int:
            score = bigram_int[bi_cand]
            cache.append(intense_section(score))
            n = next_n + 1
          else:   # extract unigram keywords
            uni_cand = line[n]
            res = check_uni(uni_cand)
            if res != None:
              cache.append(res)
            n +=1
        else:   # check the last word of the line
          uni_cand = line[n]
          res = check_uni(uni_cand)
          if res != None:
            cache.append(res)
          n +=1

    if len(cache) != 0:
      final_line.append(cache)

  final_line_with_seq = []
  # add <seq> between each line
  for i, line in enumerate(final_line):
    if i < len(final_line) - 1:
      line.append(vocab_idx['<seq>'])
    final_line_with_seq.extend(line)

  final_line_with_seq.insert(0, vocab_idx['<cls>'])
  
  return final_line_with_seq

"""mask the intensive."""
# Intensive vocab ids
intensive_negation_idx_set = [vocab_idx['<int_-3>'], vocab_idx['<int_-2>'], vocab_idx['<int_-1>'], vocab_idx['<int_0>'], vocab_idx['<negation>']]
def mask_intensive(vocab):
  if vocab in intensive_negation_idx_set or vocab == vocab_idx['<padding>']:
    return 0
  else:
    return 1

"""mask the sentence."""
# no need to intensive
no_need_intensive = [vocab_idx['<padding>'], vocab_idx['<seq>']]
def make_mask(line):
  # initialize the mask
  initial_mask = np.reshape(np.array(list(map(mask_intensive, line)), dtype=np.int32), [-1, 1])
  initial_all_mask = np.dot(initial_mask, initial_mask.T)

  # find the intensive index
  int_idx = [i for i, v in enumerate(line) if v in intensive_negation_idx_set]
  for i in int_idx:
    # change the intensive line mask
    initial_all_mask[i][i] = 1
    # change mask for the vocab behind the intensive vocab
    if i + 1 < len(line):
      if line[i + 1] not in no_need_intensive:
        initial_all_mask[i][i+1] = 1
        initial_all_mask[i+1][i] = 1
  return initial_all_mask

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
    input_mask = list(map(make_mask, sentences_idx_padded))

    features = {'input_data': sentences_idx_padded,
                'input_mask': input_mask}
    yield(features, labels)
     
def train_input_fn():
  output_types = {'input_data': tf.int32, 
                  'input_mask': tf.int32}
  output_shapes = {'input_data': [None, None],
                   'input_mask': [None, None, None]}
  
  dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_types=(output_types, tf.int32),
    output_shapes=(output_shapes, [None]))
  
  dataset = dataset.repeat(train_steps)
  return dataset

def server_input_fn():
  input_data = tf.placeholder(tf.int32, shape=[None, None], name='input_data')
  input_mask = tf.placeholder(tf.int32, shape=[None, None, None], name='input_mask')

  receive_tensors = {'input_data': input_data, 'input_mask': input_mask}
  features = {'input_data': input_data, 'input_mask': input_mask}

  return tf.estimator.export.ServingInputReceiver(features, receive_tensors)

if __name__ == '__main__':
  # for l in train_generator():
  #   print(l)
    # input()

  for data in train_input_fn():
    print(data)
    # input()

  # test_sentence = 'this is a test, i don\' know, this much admired is cool.howerve this is cool, more than cool cool cool.<br /><br />hahaha, this is haha, go back.'
  # print(process_line(test_sentence))

  # test_data = [0, 100, 200, 3, 100, 3, 2, 100, 1, 1]
  # mask = make_mask(test_data)
  # print(mask)
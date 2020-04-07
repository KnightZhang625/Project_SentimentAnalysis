# coding:utf-8

import sys
import math
import codecs
import pickle
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from preprocess_data import save_to_binary
from log import log_info as _info
from log import log_error as _error

def load_dict(path):
  """load the dictionary.
  
  Returns:
    A list consists of tuples with vocab and value.
  """
  data = []
  with codecs.open(path, 'r', 'utf-8') as file:
    for line in file:
      line = line.strip()
      if len(line) > 0:
        vocab, value = line.split('\t')[0], line.split('\t')[1]
        data.append((vocab, value))
  _info('{} contains {} vocabs.'.format(path, len(data)))
  return data

def make_dict(data, split_tag):
  vocab_dict = {}
  bi_vocab_dict = {}
  for item in data:
    vocab, value = item[0].strip(), item[1]
    if len(vocab.split(split_tag)) == 2:
      vocab = ' '.join(vocab.split(split_tag))  # remove split_tag
      bi_vocab_dict[vocab] = float(value)
    else:
      vocab_dict[vocab] = float(value)
  return vocab_dict, bi_vocab_dict

if __name__ == '__main__':
  # adj_data = load_dict(MAIN_PATH / 'data/dictionary/adj_dictionary1.11.txt')
  # adv_data = load_dict(MAIN_PATH / 'data/dictionary/adv_dictionary1.11.txt')
  # noun_data = load_dict(MAIN_PATH / 'data/dictionary/noun_dictionary1.11.txt')
  # verb_data = load_dict(MAIN_PATH / 'data/dictionary/verb_dictionary1.11.txt')
  # int_data = load_dict(MAIN_PATH / 'data/dictionary/int_dictionary1.11.txt')
  # all_data = adj_data + adv_data + noun_data + verb_data

  # vocab_dict, bi_vocab_dict = make_dict(all_data, split_tag='-')
  # int_dict, bi_int_dict = make_dict(int_data, split_tag='_')

  # save_to_binary(vocab_dict, MAIN_PATH / 'data/dictionary_binary_new/keywords.bin')
  # save_to_binary(bi_vocab_dict, MAIN_PATH / 'data/dictionary_binary_new/bi_keywords.bin')
  # save_to_binary(int_dict, MAIN_PATH / 'data/dictionary_binary_new/int.bin')
  # save_to_binary(bi_int_dict, MAIN_PATH / 'data/dictionary_binary_new/bi_int.bin')
  
  supplement_vocab = ['<cls>', '<padding>', '<seq>', '<int_-3>', '<int_-2>', '<int_-1>', '<int_0>', '<negation>']
  keywords_vocab = [int(i) for i in range(-5, 6)]
  final_vocab = keywords_vocab + supplement_vocab

  final_dict = {}
  for i, v in enumerate(final_vocab):
    final_dict[v] = i
  print(final_dict)
  save_to_binary(final_dict, MAIN_PATH / 'data/dictionary_binary_new/vocab.bin')
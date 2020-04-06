# coding:utf-8

import sys
import codecs
import pickle
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from preprocess_data import make_dict as mk
from preprocess_data import save_to_binary

from nltk.stem import PorterStemmer 
ps = PorterStemmer()

# load the dictionary
def load_dict(path):
  with codecs.open(path, 'rb') as file:
    data = pickle.load(file)
  return data

def purify_each_item(vocab):
  vocab = vocab.strip()
  return [vocab] if ps.stem(vocab) == vocab else [vocab, ps.stem(vocab)]

def process_each_data(vocab_data):
  part_dict = []    # this is for making dict
  part_dict_bi = [] # this is for searching bigram
  for vocab in vocab_data:
    # add bigram vocab
    if len(vocab.split('-')) == 2:
      part_dict_bi.append(vocab.replace('-', ' '))
      part_dict.append(vocab.replace('-', ' '))
    else:
      part_dict.extend(purify_each_item(vocab))
  return part_dict, part_dict_bi

def make_dict(dict_path_set):
  unigram_vocab = []
  bigram_vocab = []
  
  for path in dict_path_set:
    with codecs.open(path, 'rb') as file:
      vocab_data = pickle.load(file)
      part_dict, part_dict_bi = process_each_data(vocab_data)
      unigram_vocab.extend(part_dict)
      bigram_vocab.extend(part_dict_bi)
  
  return unigram_vocab, bigram_vocab

def make_dict_for_int(dict_path):
  new_dict = {}
  new_dict_bi = []
  with codecs.open(dict_path, 'rb') as file:
    data = pickle.load(file)
  for vocab, value in data.items():
    if len(vocab.split('_')) > 1:
      vocab = vocab.replace('_', ' ')
      new_dict_bi.append(vocab)
      new_dict[vocab] = value
    else:
      new_dict[vocab] = value
  return new_dict, new_dict_bi
 
if __name__ == '__main__':
  prefix_path = MAIN_PATH / 'data/dictionary_binary'
  # dict_path_set = [prefix_path / 'adj.bin',
  #                  prefix_path / 'adv.bin',
  #                  prefix_path / 'noun.bin',
  #                  prefix_path / 'verb.bin']
  
  # unigram_vocab, bigram_vocab = make_dict(dict_path_set)
  # vocab_idx = mk(unigram_vocab, False)
  # save_to_binary(vocab_idx, prefix_path / 'vocab_idx_new.bin')
  # save_to_binary(bigram_vocab, prefix_path / 'bigram.bin')

  new_dict, new_dict_bi = make_dict_for_int(prefix_path / 'int.bin')
  save_to_binary(new_dict, prefix_path / 'int_new.bin')
  save_to_binary(new_dict_bi, prefix_path / 'bigram_int.bin' )
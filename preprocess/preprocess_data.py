# coding:utf-8

import sys
import codecs
import pickle
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from log import log_info as _info
from log import log_error as _error
from log import print_process as _process

def save_to_binary(data, save_path, replace=False):
  """convert the data to binary file and save.
  
  Args:
    data: object, the original file.
    save_path: str, the absolute path to save the data.
    replace: boolean, Whether to replace the file when the file exits.
  """
  # change the str path to PosixPath
  save_path = Path(save_path)
  
  # check the file exits or not
  if save_path.is_file():
    if not replace:
      _error('{} already exits.'.format(save_path), head='ERROR')
      raise FileExistsError
    else:
      _info('{} already exits, replaced.'.format(save_path))
  
  with codecs.open(save_path, 'wb') as file:
    pickle.dump(data, file)

def read_dictionary(path, split_tag='\t', need_score=False):
  """read the vocab from the dictionary.
  
  Args:
    path: str, the absolute path of the dictionary.
    split_tag: str, the splitting tag for each line.
    need_score: boolean, need the score behind the vocab or not.
  
  Returns:
    a list(or dict) contains the whole vocabulary.
  """
  if not need_score:
    vocab_set = []
  else:
    vocab_set = {}

  with codecs.open(path, 'r', 'utf-8') as file:
    for line in file:
      line = line.strip()
      if len(line) > 0:
        line_split = line.split(split_tag)
        if not need_score:
          vocab = line_split[0]
          vocab_set.append(vocab)
        else:
          vocab, score = line_split[0], float(line_split[1])
          vocab_set[vocab] = score

  if not need_score:
    _info('{} contains {} vocabs.\n Some vocabs looks like {}.\n'.format(
      path, len(vocab_set), vocab_set[:5] + vocab_set[-5:]))
  else:
    _info('{} contains {} vocabs.\n Some vocabs looks like {}.\n'.format(
      path, len(vocab_set), list(vocab_set.keys())[:5] + list(vocab_set.keys())[-5:]))
  return vocab_set

def read_twitter_data(path, split_tag='\t', polarity=None):
  """read the twitter data from the given path."""
  data = []
  with codecs.open(path, 'r', 'utf-8') as file:
    for line in file:
      line = line.strip()
      if len(line) > 0:
        if split_tag is not None:
          line_strip = line.split(split_tag)
          id_, tag, sentence = line_strip[0], line_strip[1], line_strip[2]
          data.append((id_, tag, sentence))
        else:
          data.append((polarity, line))
  _info('{} contains {} data.'.format(path, len(data)))
  if split_tag is not None:
    return data
  else:
    return data[0]

def make_dict(path_set, path_or_not=True):
  if path_or_not:
    vocab = []
    for path in path_set:
      with codecs.open(path, 'rb') as file:
        data = pickle.load(file)
      for v in data:
        vocab.append(v)
  else:
    vocab = path_set
  
  supplement_vocab = ['<cls>', '<padding>', '<seq>', '<int_-3>', '<int_-2>', '<int_-1>', '<int_0>', '<negation>']
  vocab = supplement_vocab + vocab

  vocab_idx = {}
  for i, v in enumerate(vocab):
    vocab_idx[v] = i
  
  return vocab_idx

if __name__ == '__main__':
  # # load the dictionary
  # adj_vocabs = read_dictionary('../data/dictionary/adj_dictionary1.11.txt')
  # adv_vocabs = read_dictionary('../data/dictionary/adv_dictionary1.11.txt')
  # intensive_vocabs = read_dictionary('../data/dictionary/int_dictionary1.11.txt', need_score=True)
  # noun_vocabs = read_dictionary('../data/dictionary/noun_dictionary1.11.txt')
  # verb_vocabs = read_dictionary('../data/dictionary/verb_dictionary1.11.txt')
  # negation_vocabs = read_dictionary('../data/dictionary/negation_dictionary.txt')

  # # save the vocabs to binary file
  # save_path_prefix = MAIN_PATH / 'data/dictionary_binary'
  # save_to_binary(adj_vocabs, save_path_prefix / 'adj.bin')
  # save_to_binary(adv_vocabs, save_path_prefix / 'adv.bin')
  # save_to_binary(intensive_vocabs, save_path_prefix / 'int.bin')
  # save_to_binary(noun_vocabs, save_path_prefix / 'noun.bin')
  # save_to_binary(verb_vocabs, save_path_prefix / 'verb.bin')
  # save_to_binary(negation_vocabs, save_path_prefix / 'neg.bin')

  # read the data and save
  data_path_list = (MAIN_PATH / 'data/2017_English_final/GOLD/Subtask_A/').rglob('twitter-*')
  save_path_prefix = MAIN_PATH / 'data/data_binary'
  for path in data_path_list:
    data = read_twitter_data(path)
    file_name = str(path).split('/')[-1].replace('.txt', '')
    save_to_binary(data, save_path_prefix / '{}.bin'.format(file_name))\
  
  # data_path_list_pos = (MAIN_PATH / 'data/Stanford_Data/test/pos').rglob('*.txt')
  data_path_list_pos = (MAIN_PATH / 'data/Stanford_Data/test/pos').rglob('[0-9]*.txt')
  # data_path_list_neg = (MAIN_PATH / 'data/Stanford_Data/test/neg').rglob('*.txt')
  data_path_list_neg = (MAIN_PATH / 'data/Stanford_Data/test/neg').rglob('[0-9]*.txt')
  save_path_prefix = MAIN_PATH / 'data/Stanford_Data_binary'
  
  data_positve = []
  data_negative = []
  
  for path in data_path_list_pos:
    data = read_twitter_data(path, None, 1)
    data_positve.append(data)
  save_to_binary(data_positve, save_path_prefix / 'test_train_pos.bin')

  for path in data_path_list_neg:
    data = read_twitter_data(path, None, 0)
    data_negative.append(data)
  save_to_binary(data_negative, save_path_prefix / 'test_train_neg.bin')

  # # build vocab_idx
  # prefix_path = MAIN_PATH / 'data/dictionary_binary'
  # dict_path_set = [prefix_path / 'adj.bin',
  #                  prefix_path / 'adv.bin',
  #                  prefix_path / 'noun.bin',
  #                  prefix_path / 'verb.bin']
  # vocab_idx = make_dict(dict_path_set)
  # save_to_binary(vocab_idx, prefix_path / 'vocab_idx.bin')
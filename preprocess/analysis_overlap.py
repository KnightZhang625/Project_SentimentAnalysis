# coding:utf-8

import re
import sys
import random
import codecs
import pickle
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from log import log_info as _info
from log import log_error as _error
from log import print_process as _process

def load_binary(path):
  """load the binary data."""
  with codecs.open(path, 'rb') as file:
    data = pickle.load(file)
  return data

def read_data(dir_path):
  """read the twitter data."""
  data_all = []
  data_path = dir_path.rglob('*.bin')
  for path in data_path:
    with codecs.open(path, 'rb') as file:
      data = pickle.load(file)
      data_all.extend(data)
  return data_all

# ELIMINATE_PUNCTUATION = '[’"#$%&()*+/:;<=>@[\\]^_`{|}~。（）：＊※，·… 、？！\n👍～《 》「」”]+'
ELIMINATE_PUNCTUATION = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~。（）：＊※，·… 、？！\n👍～《 》「」”]+'

def check_overlap(data, dic_set, single_data_or_not=True):
  match_number = [0 for _ in data]
  match_per_length = 0
  total_length = 0

  for idx, line in enumerate(data):
    single_length = 0
    match_words = []
    if single_data_or_not:
      line = line[0]
      sentence = re.sub(ELIMINATE_PUNCTUATION, ' ', line[1])
    else:
      sentence = re.sub(ELIMINATE_PUNCTUATION, ' ', line[2])
    sentence_split = sentence.split(' ')
    for vocab in sentence_split:
      if vocab in dic_set:
        match_words.append(vocab)
        single_length += 1
        match_per_length += 1
        match_number[idx] = 1
    # _info(line[0])
    # print(single_length / len(sentence_split))
    # print(line)
    # print(match_words)
    # input()
    single_length = 0
    total_length += len(sentence_split)

  _info('TOTAL DATA: {} MATCH: {} TOTAL_MATCH: {}'.format(len(data), sum(match_number), match_per_length / total_length))

if __name__ == '__main__':
  # load the dic
  dir_path_prefix = MAIN_PATH / 'data/dictionary_binary'
  adj_vocab = load_binary(dir_path_prefix / 'adj.bin')
  adv_vocab = load_binary(dir_path_prefix / 'adv.bin')
  int_vocab = load_binary(dir_path_prefix / 'int.bin')
  neg_vocab = load_binary(dir_path_prefix / 'neg.bin')
  noun_vocab = load_binary(dir_path_prefix / 'noun.bin')
  verb_vocab = load_binary(dir_path_prefix / 'verb.bin')

  # scores = []
  # for _, value in int_vocab.items():
  #   scores.append(value)
  # print(len(scores))
  # print(min(scores))
  # print(max(scores))

  # # make hash
  # dic_hash = {}
  # for dic in [adj_vocab, adv_vocab, neg_vocab, noun_vocab, verb_vocab]:
  #   for v in dic:
  #     dic_hash[v] = 0

  # # load the data
  # dir_path = MAIN_PATH / 'data/Stanford_Data_binary'
  # data_all = read_data(dir_path)

  # # check overlap
  # random.shuffle(data_all)
  # check_overlap(data_all, dic_hash)
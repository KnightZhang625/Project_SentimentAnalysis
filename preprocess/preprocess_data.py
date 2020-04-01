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
  if save_path.is_file() and not replace:
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

if __name__ == '__main__':
  # load the dictionary
  adj_vocabs = read_dictionary('../data/dictionary/adj_dictionary1.11.txt')
  adv_vocabs = read_dictionary('../data/dictionary/adv_dictionary1.11.txt')
  intensive_vocabs = read_dictionary('../data/dictionary/int_dictionary1.11.txt', need_score=True)
  noun_vocabs = read_dictionary('../data/dictionary/noun_dictionary1.11.txt')
  verb_vocabs = read_dictionary('../data/dictionary/verb_dictionary1.11.txt')
  negation_vocabs = read_dictionary('../data/dictionary/negation_dictionary.txt')

  # save the vocabs to binary file
  save_path_prefix = MAIN_PATH / 'data/dictionary_binary'
  save_to_binary(adj_vocabs, save_path_prefix / 'adj.bin')
  save_to_binary(adv_vocabs, save_path_prefix / 'adv.bin')
  save_to_binary(intensive_vocabs, save_path_prefix / 'int.bin')
  save_to_binary(noun_vocabs, save_path_prefix / 'noun.bin')
  save_to_binary(verb_vocabs, save_path_prefix / 'verb.bin')
  save_to_binary(negation_vocabs, save_path_prefix / 'neg.bin')
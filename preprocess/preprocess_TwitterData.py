# coding:utf-8

import sys
import codecs
import pickle
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from log import log_info as _info
from log import log_error as _error

def load_data(path):
  with codecs.open(path, 'r', 'utf-8') as file:
    for line in file:
      print(line)


if __name__ == '__main__':
  dir_path = MAIN_PATH / 'data/2017_English_final/GOLD/Subtask_A'

  train_data_path_list = Path(dir_path).rglob('*train*.txt')
  dev_data_path_list = Path(dir_path).rglob('*dev*.txt')
  print(list(dev_data_path_list))
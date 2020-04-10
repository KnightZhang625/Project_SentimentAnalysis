# coding:utf-8

from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent

# data
TRAIN_POS_DATA_PATH = MAIN_PATH / 'data/Stanford_Data_binary/train_pos.bin'
TRAIN_NEG_DATA_PATH = MAIN_PATH / 'data/Stanford_Data_binary/train_neg.bin'
VOCAB_IDX_PATH = MAIN_PATH / 'data/vocab_idx.bin'
INT_PATH = MAIN_PATH / 'data/dictionary_binary_new/int.bin'
NEG_PATH = MAIN_PATH / 'data/dictionary_binary/neg.bin'
KEYWORDS_PATH = MAIN_PATH / 'data/dictionary_binary_new/keywords.bin'
BIGRAM_PATH = MAIN_PATH / 'data/dictionary_binary_new/bi_keywords.bin'
BIGRAM_INT_PATH = MAIN_PATH / 'data/dictionary_binary_new/bi_int.bin'

# model path
save_model_path = 'models/'
pb_model_path = 'pb_models/'
keep_checkpoint_max = 1
save_checkpoints_steps = 1000 

# global
batch_size = 2
train_steps = 1
print_info_interval = 10
learning_rate = 1e-3
lr_limit = 1e-4
colocate_gradients_with_ops = True
random_mask_prob = 0.5
enable_vae = False

# Bert
class BertEncoderConfig(object):
  hidden_dropout_prob = 0.2
  attention_dropout_prob = 0.2

  vocab_size = 29258
  num_classes = 2
  embedding_size = 30
  max_positional_embeddings = 2000
  hidden_size = 30
  num_hidden_layers = 2
  num_attention_heads = 2
  intermediate_size = 320
  intermediate_before_final_output_size = 320

  initializer_range = 0.02
  hidden_act = 'gelu'
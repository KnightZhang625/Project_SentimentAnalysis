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
batch_size = 32
train_steps = 10000
print_info_interval = 10
learning_rate = 2e-5
lr_limit = 2e-5
colocate_gradients_with_ops = True
random_mask_prob = 0.5
# enable_vae = False

# Bert
class BertEncoderConfig(object):
  hidden_dropout_prob = 0.1
  attention_dropout_prob = 0.1

  vocab_size = 30522
  num_classes = 2
  embedding_size = 256
  max_positional_embeddings = 512
  hidden_size = 256
  num_hidden_layers = 6
  num_attention_heads = 4
  intermediate_size = 1024
  # intermediate_before_final_output_size = 320

  initializer_range = 0.02
  hidden_act = 'gelu'
import random
from collections import defaultdict
from models.bert import modeling_bert, configuration_bert
from models.electra import modeling_electra, configuration_electra

# Pretrained_Model PARAMS
BERT_MODEL_PARAMS = defaultdict(
  pretrained_config = configuration_bert.BertConfig,
  pretrained_model = modeling_bert.BertModel
)

ELECTRA_MODEL_PARAMS = defaultdict(
  pretrained_config = configuration_electra.ElectraConfig,
  pretrained_model = modeling_electra.ElectraModel
)

# DATASET
UBUNTU_PARAMS = defaultdict(
  evaluate_candidates_num=10,
  recall_k_list=[1,2,5,10],
  evaluate_data_type="test",
  language="english",
  eval_batch_size=10,
)

DOUBAN_PARAMS = defaultdict(
  evaluate_candidates_num=10,
  recall_k_list=[1, 2, 5,10],
  evaluate_data_type="test",
  language="chinese",
  max_utt_len=5,
)

ECOMMERCE_PARAMS = defaultdict(
  evaluate_candidates_num=10,
  recall_k_list=[1, 2, 5,10],
  evaluate_data_type="test",
  language="chinese",
  max_utt_len=5,
)

# MULTI-TASK PARAMS
INSERTION_PARAMS = defaultdict(
  do_sent_insertion=True,  # True or False for jude
)
DELETION_PARAMS = defaultdict(
  do_sent_deletion=True,  # True or False for jude
)
SEARCH_PARAMS = defaultdict(
  do_sent_search=True,  # True or False for jude
)

BASE_PARAMS = defaultdict(
  # lambda: None,  # Set default value to None.
  # GPU params
  gpu_ids=[0], # [0,1,2,3]

  # Input params
  train_batch_size=4, # 32
  eval_batch_size=250, # 1000
  virtual_batch_size=32,

  # Training BERT params
  learning_rate=3e-05, # 2e-05
  # learning_rate=2e-5,

  dropout_keep_prob=0.8,
  num_epochs=5,
  max_gradient_norm=5,
  adam_epsilon=1e-8,
  weight_decay=0.0,
  warmup_steps=0,
  optimizer_type="AdamW",  # Adam, AdamW

  pad_idx=0,
  max_position_embeddings=512,
  num_hidden_layers=12,
  num_attention_heads=12,
  intermediate_size=3072,
  bert_hidden_dim=768,
  attention_probs_dropout_prob=0.1,
  layer_norm_eps=1e-12,

  # Train Model Config
  do_shuffle_ressel=False,
  pca_visualization=False,

  task_name="ubuntu",
  do_bert=True,
  do_eot=True,

  do_response_selection=True,
  do_adversarial_test=True,

  auxiliary_loss_type="sigmoid", # softmax or sigmoid
  do_sent_insertion=False, # True or False for jude
  do_sent_deletion=False, # True or False for jude
  do_sent_search=False, # True or False for jude

  max_sequence_len=512,
  res_sel_loss_ratio=1.0,
  ins_loss_ratio=1.0, # 0.01, 0.1, 0.5, 1,
  del_loss_ratio=1.0, # 0.01, 0.1, 0.5, 1,
  srch_loss_ratio=1.0, # 0.01, 0.1, 0.5, 1,
  max_utt_len=5, # ubuntu : 5

  save_dirpath='checkpoints/', # /path/to/checkpoints

  load_pthpath="",
  cpu_workers=0,
  tensorboard_step=100,
  evaluate_print_step=100,
  random_seed= random.sample(range(1000, 10000), 1)[0], # 3143
)

POST_PARAMS = BASE_PARAMS.copy()
POST_PARAMS.update(
  model_type="bert_post"
)

ELECTRA_POST_PARAMS = BASE_PARAMS.copy()
ELECTRA_POST_PARAMS.update(
  model_type="electra_post"
)

BASE_EOT_PARAMS = BASE_PARAMS.copy()
BASE_EOT_PARAMS.update(
  model_type="bert_base_eot"
)

BERT_POST_TRAINING_PARAMS = BASE_PARAMS.copy()
BERT_POST_TRAINING_PARAMS.update(
  num_epochs=3, # duplicate 10 * 3 (ubuntu case : 2 epochs)
  gpu_ids=[0,1,2,3],

  # Input params
  train_batch_size=96, # 96 for jude
  virtual_batch_size=384, # 384 for jude
  tensorboard_step=100,

  checkpoint_save_step=2500, # virtual_batch -> 10000 step
)

ELECTRA_POST_TRAINING_PARAMS = BASE_PARAMS.copy()
ELECTRA_POST_TRAINING_PARAMS.update(
  num_epochs=20, # duplicate 10 * 2 epoch
  gpu_ids=[0,1,2,3], # [0,1,2,3] for jude

  # Input params
  train_batch_size=8, # 64 for jude
  virtual_batch_size=256, # 256 for jude
  tensorboard_step=100,

  optimizer_type="Adam",

  checkpoint_save_step=2500, # virtual_batch -> 10000 step
  electra_gen_ckpt_path="electra-base-chinese-gen-pytorch_model.bin", # electra-base-gen-pytorch_model.bin
  electra_gen_config="electra-base-chinese-gen", # electra-base-gen
  max_masked_tok_num=70,
  electra_disc_ratio=50,
)

ELECTRA_NSP_POST_TRAINING_PARAMS = BASE_PARAMS.copy()
ELECTRA_NSP_POST_TRAINING_PARAMS.update(
  num_epochs=3, # duplicate 10 * 2 epoch
  gpu_ids=[0,1,2,3], # [0,1,2,3] for jude

  # Input params
  train_batch_size=96, # 64 for jude
  virtual_batch_size=384, # 256 for jude
  tensorboard_step=100,

  optimizer_type="Adam",
  learning_rate=3e-05,  # 2e-05

  checkpoint_save_step=2500, # virtual_batch -> 10000 step
  electra_disc_ratio=50,
)
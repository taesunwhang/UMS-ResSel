import os
import pickle
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from models.bert import tokenization_bert


class ElectraPostTrainingDataset(Dataset):
  """
  A full representation of VisDial v1.0 (train/val/test) dataset. According
  to the appropriate split, it returns dictionary of question, image,
  history, ground truth answer, answer options, dense annotations etc.
  """

  def __init__(self, hparams, split=""):
    super().__init__()

    self.hparams = hparams
    self._input_examples = []

    bert_pretrained_dir = os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained)
    print(bert_pretrained_dir)
    self._tokenizer = tokenization_bert.BertTokenizer(
      vocab_file=os.path.join(bert_pretrained_dir, "%s-vocab.txt" % self.hparams.bert_pretrained))
    self._vocab = self._tokenizer.vocab

    with open(os.path.join(hparams.data_dir, "%s_electra_post_training.pkl" % hparams.task_name), "rb") as pkl_handle:
      while True:
        try:
          self._input_examples.append(pickle.load(pkl_handle))
          if len(self._input_examples) % 100000 == 0:
            print("%d examples has been loaded!" % len(self._input_examples))
        except EOFError:
          break

    print("total post-training examples : %d" % len(self._input_examples))

  def __len__(self):
    return len(self._input_examples)

  def __getitem__(self, index):
    # Get Input Examples
    """
    InputExamples
      input_ids
      input_mask
      token_type_ids
    """
    example = self._input_examples[index]

    anno_input_ids, anno_masked_lm_labels = self._anno_mask_inputs(example.input_ids, example.input_mask)
    curr_features = dict()
    curr_features["masked_input_ids"] = torch.tensor(anno_input_ids).long()
    curr_features["input_ids"] = torch.tensor(example.input_ids).long()
    curr_features["attention_mask"] = torch.tensor(example.input_mask).long()
    curr_features["token_type_ids"] = torch.tensor(example.token_type_ids).long()
    curr_features["masked_lm_labels"] = torch.tensor(anno_masked_lm_labels).long()

    return curr_features

  # dynamic masking
  def _anno_mask_inputs(self, input_ids, input_mask, max_seq_len=512):
    # masked_lm_ids -> labels
    anno_masked_lm_labels = [-1] * max_seq_len
    anno_input_ids = input_ids.copy()

    curr_seq_len = sum(input_mask)
    masked_tok_num = min(self.hparams.max_masked_tok_num, int((curr_seq_len - 2) * 0.15))
    masked_tok_pos = random.sample(list(range(curr_seq_len)), masked_tok_num)  # without cls and sep

    for pos in masked_tok_pos:
      if anno_input_ids[pos] in [self._vocab["[CLS]"], self._vocab["[SEP]"], self._vocab["[UNK]"]]:
        continue
      anno_input_ids[pos] = self._vocab["[MASK]"]
      anno_masked_lm_labels[pos] = input_ids[pos]

    return anno_input_ids, anno_masked_lm_labels

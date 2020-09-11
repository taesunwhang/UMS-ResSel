import os
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset

class BertPostTrainingDataset(Dataset):
  """
  A full representation of VisDial v1.0 (train/val/test) dataset. According
  to the appropriate split, it returns dictionary of question, image,
  history, ground truth answer, answer options, dense annotations etc.
  """

  def __init__(
      self,
      hparams,
      split: str = "",
  ):
    super().__init__()

    self.hparams = hparams
    self.split = split

    with h5py.File(os.path.join(self.hparams.data_dir, "%s_post_training.hdf5" % self.hparams.task_name), "r") as features_hdf:
      self.feature_keys = list(features_hdf.keys())
      self.num_instances = np.array(features_hdf.get("next_sentence_labels")).shape[0]
    print("total %s examples : %d" % (split, self.num_instances))

  def __len__(self):
    return self.num_instances

  def __getitem__(self, index):
    # Get Input Examples
    """
    InputExamples
      self.utterances = utterances
      self.response = response
      self.label
    """
    features = self._read_hdf_features(index)
    anno_masked_lm_labels = self._anno_mask_inputs(features["masked_lm_ids"], features["masked_lm_positions"])
    curr_features = dict()
    for feat_key in features.keys():
      curr_features[feat_key] = torch.tensor(features[feat_key]).long()
    curr_features["masked_lm_labels"] = torch.tensor(anno_masked_lm_labels).long()

    return curr_features

  def _read_hdf_features(self, index):
    features = {}
    with h5py.File(os.path.join(self.hparams.data_dir, "%s_post_training.hdf5" % self.hparams.task_name), "r") as features_hdf:
      for f_key in self.feature_keys:
        features[f_key] = features_hdf[f_key][index]

    return features

  def _anno_mask_inputs(self, masked_lm_ids, masked_lm_positions, max_seq_len=512):
    # masked_lm_ids -> labels
    anno_masked_lm_labels = [-1] * max_seq_len

    for pos, label in zip(masked_lm_positions, masked_lm_ids):
      if pos == 0: continue
      anno_masked_lm_labels[pos] = label

    return anno_masked_lm_labels
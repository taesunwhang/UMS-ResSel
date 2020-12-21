# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

sys.path.append(os.getcwd())
import collections
import random
import argparse
import h5py
import numpy as np
from tqdm import tqdm

from models.bert import tokenization_bert


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


class CreateBertPretrainingData(object):
  def __init__(self, args):
    self.args = args
    self._bert_tokenizer_init(args.special_tok, args.bert_pretrained)

  def _bert_tokenizer_init(self, special_tok, bert_pretrained):
    bert_pretrained_dir = os.path.join("./resources", bert_pretrained)
    vocab_file_path = "%s-vocab.txt" % bert_pretrained

    self._bert_tokenizer = tokenization_bert.BertTokenizer(
      vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path))
    self._bert_tokenizer.add_tokens([special_tok])

    print("BERT tokenizer init completes")

  def _add_special_tokens(self, tokens, special_tok="[EOT]"):
    tokens = tokens + [special_tok]
    return tokens

  def create_training_instances(self, input_file, max_seq_length,
                                dupe_factor, short_seq_prob, masked_lm_prob,
                                max_predictions_per_seq, rng, special_tok=None):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    document_cnt = 0
    with open(input_file, "r", encoding="utf=8") as fr_handle:
      for line in tqdm(fr_handle):
        line = line.strip()

        # Empty lines are used as document delimiters
        if len(line) == 0:
          all_documents.append([])
          document_cnt += 1
          if document_cnt % 50000 == 0:
            print("%d documents have been tokenized!" % document_cnt)

        tokens = self._bert_tokenizer.tokenize(line)
        if special_tok and len(tokens) > 0:
          tokens = self._add_special_tokens(tokens, special_tok)  # special tok per sentence

        if tokens:
          all_documents[-1].append(tokens)
    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(self._bert_tokenizer.vocab.keys())

    self.feature_keys = ["input_ids", "attention_mask", "token_type_ids",
                         "masked_lm_positions", "masked_lm_ids", "next_sentence_labels"]

    print("Total number of documents : %d" % len(all_documents))
    hf = h5py.File(self.args.output_file, "w")
    for d in range(dupe_factor):
      rng.shuffle(all_documents)

      self.all_doc_feat_dict = dict()
      for feat_key in self.feature_keys:
        self.all_doc_feat_dict[feat_key] = []

      for document_index in tqdm(range(len(all_documents))):
        instances = self.create_instances_from_document(
          all_documents, document_index, max_seq_length, short_seq_prob,
          masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        self.instance_to_example_feature(instances, self.args.max_seq_length, self.args.max_predictions_per_seq)
      self.build_h5_data(hf, d_idx=d),
      print("Current Dupe Factor : %d" % (d + 1))
    print("Pretraining Data Creation Completes!")
    hf.close()

  def build_h5_data(self, hf, d_idx=0):
    """
      features["input_ids"] = torch.tensor(input_ids).long()
      features["attention_mask"] = torch.tensor(input_mask).long()
      features["token_type_ids"] = torch.tensor(segment_ids).long()
      features["masked_lm_positions"] = torch.tensor(masked_lm_positions).long()
      features["masked_lm_ids"] = torch.tensor(masked_lm_ids).long() # masked_lm_ids
      features["next_sentence_labels"] = torch.tensor([next_sentence_label]).long()
    """

    h5_key_dict = {}
    print("Number of documents features", len(self.all_doc_feat_dict["next_sentence_labels"]))
    if d_idx == 0:
      for feat_key in self.feature_keys:
        key_size = [len(self.all_doc_feat_dict["next_sentence_labels"])] + [len(self.all_doc_feat_dict[feat_key][0])]
        h5_key_dict[feat_key] = hf.create_dataset(feat_key, tuple(key_size), dtype='i8', chunks=True,
                                                  maxshape=(None, tuple(key_size)[1]),
                                                  data=np.array(self.all_doc_feat_dict[feat_key]))
    else:
      for feat_key in self.feature_keys:
        hf[feat_key].resize((hf[feat_key].shape[0] + len(self.all_doc_feat_dict["next_sentence_labels"])), axis=0)
        hf[feat_key][-len(self.all_doc_feat_dict["next_sentence_labels"]):] = np.array(self.all_doc_feat_dict[feat_key])

  def create_instances_from_document(self, all_documents, document_index, max_seq_length, short_seq_prob,
                                     masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
      target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0

    while i < len(document):
      segment = document[i]
      current_chunk.append(segment)
      current_length += len(segment)
      if i == len(document) - 1 or current_length >= target_seq_length:
        if current_chunk:
          # `a_end` is how many segments from `current_chunk` go into the `A`
          # (first) sentence.
          a_end = 1
          if len(current_chunk) >= 2:
            a_end = rng.randint(1, len(current_chunk) - 1)

          tokens_a = []
          for j in range(a_end):
            tokens_a.extend(current_chunk[j])

          tokens_b = []
          # Random next
          is_random_next = False
          if len(current_chunk) == 1 or rng.random() < 0.5:
            is_random_next = True
            target_b_length = target_seq_length - len(tokens_a)

            # This should rarely go for more than one iteration for large
            # corpora. However, just to be careful, we try to make sure that
            # the random document is not the same as the document
            # we're processing.
            for _ in range(10):
              random_document_index = rng.randint(0, len(all_documents) - 1)
              if random_document_index != document_index:
                break

            random_document = all_documents[random_document_index]
            random_start = rng.randint(0, len(random_document) - 1)
            for j in range(random_start, len(random_document)):
              tokens_b.extend(random_document[j])
              if len(tokens_b) >= target_b_length:
                break
            # We didn't actually use these segments so we "put them back" so
            # they don't go to waste.
            num_unused_segments = len(current_chunk) - a_end
            i -= num_unused_segments
          # Actual next
          else:
            is_random_next = False
            for j in range(a_end, len(current_chunk)):
              tokens_b.extend(current_chunk[j])
          self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

          assert len(tokens_a) >= 1
          assert len(tokens_b) >= 1

          tokens = []
          segment_ids = []
          tokens.append("[CLS]")
          segment_ids.append(0)
          for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

          tokens.append("[SEP]")
          segment_ids.append(0)

          for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
          tokens.append("[SEP]")
          segment_ids.append(1)

          (tokens, masked_lm_positions, masked_lm_labels) = self.create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
          instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
          instances.append(instance)

        current_chunk = []
        current_length = 0
      i += 1

    return instances

  def create_masked_lm_predictions(self, tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
      if token == "[CLS]" or token == "[SEP]":
        continue
      # Whole Word Masking means that if we mask all of the wordpiecesdk
      # corresponding to an original word. When a word has been split into
      # WordPieces, the first token does not have any marker and any subsequence
      # tokens are prefixed with ##. So whenever we see the ## token, we
      # append it to the previous set of word indexes.
      #
      # Note that Whole Word Masking does *not* change the training code
      # at all -- we still predict each WordPiece independently, softmaxed
      # over the entire vocabulary.
      if (self.args.do_whole_word_mask and len(cand_indexes) >= 1 and
          token.startswith("##")):
        cand_indexes[-1].append(i)
      else:
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
      if len(masked_lms) >= num_to_predict:
        break
      # If adding a whole-word mask would exceed the maximum number of
      # predictions, then just skip this candidate.
      if len(masked_lms) + len(index_set) > num_to_predict:
        continue
      is_any_index_covered = False
      for index in index_set:
        if index in covered_indexes:
          is_any_index_covered = True
          break
      if is_any_index_covered:
        continue
      for index in index_set:
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
          masked_token = "[MASK]"
        else:
          # 10% of the time, keep original
          if rng.random() < 0.5:
            masked_token = tokens[index]
          # 10% of the time, replace with random word
          else:
            masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
      masked_lm_positions.append(p.index)
      masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

  def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
      total_length = len(tokens_a) + len(tokens_b)
      if total_length <= max_num_tokens:
        break

      trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
      assert len(trunc_tokens) >= 1

      # We want to sometimes truncate from the front and sometimes from the
      # back to add more randomness and avoid biases.
      if rng.random() < 0.5:
        del trunc_tokens[0]
      else:
        trunc_tokens.pop()

  def instance_to_example_feature(self, instances, max_seq_length, max_predictions_per_seq):
    for instance in instances:
      input_ids = self._bert_tokenizer.convert_tokens_to_ids(instance.tokens)
      input_mask = [1] * len(input_ids)
      segment_ids = list(instance.segment_ids)

      assert len(input_ids) <= max_seq_length

      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      masked_lm_positions = list(instance.masked_lm_positions)
      masked_lm_ids = self._bert_tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
      masked_lm_weights = [1.0] * len(masked_lm_ids)

      while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

      next_sentence_label = 1 if instance.is_random_next else 0

      self.all_doc_feat_dict["input_ids"].append(input_ids)
      self.all_doc_feat_dict["attention_mask"].append(input_mask)
      self.all_doc_feat_dict["token_type_ids"].append(segment_ids)
      self.all_doc_feat_dict["masked_lm_positions"].append(masked_lm_positions)
      self.all_doc_feat_dict["masked_lm_ids"].append(masked_lm_ids)
      self.all_doc_feat_dict["next_sentence_labels"].append([next_sentence_label])


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(description="Bert / Create Pretraining Data")
  arg_parser.add_argument("--input_file", dest="input_file", type=str, required=True,
                          help="Input raw text file (or comma-separated list of files). "
                               "e.g., ./data/ubuntu_corpus_v1/ubuntu_post_training.txt")
  arg_parser.add_argument("--output_file", dest="output_file", type=str, required=True,
                          help="Output example pkl. e.g., ./data/ubuntu_corpus_v1/ubuntu_post_training.hdf5")
  arg_parser.add_argument("--bert_pretrained", dest="bert_pretrained", type=str, required=True,
                          help="bert-base-wwm-chinese")
  arg_parser.add_argument("--do_lower_case", dest="do_lower_case", type=bool, default=True,
                          help="Whether to lower case the input text. Should be True for uncased.")
  arg_parser.add_argument("--do_whole_word_mask", dest="do_whole_word_mask", type=bool, default=True,
                          help="Whether to use whole word masking rather than per-WordPiece masking.")
  arg_parser.add_argument("--max_seq_length", dest="max_seq_length", type=int, default=512,
                          help="Maximum sequence length.")
  arg_parser.add_argument("--max_predictions_per_seq", dest="max_predictions_per_seq", type=int, default=70,
                          help="Maximum number of masked LM predictions per sequence.")
  arg_parser.add_argument("--random_seed", dest="random_seed", type=int, default=12345,
                          help="Random seed for data generation.")
  arg_parser.add_argument("--dupe_factor", dest="dupe_factor", type=int, default=10,
                          help="Number of times to duplicate the input data (with different masks).")
  arg_parser.add_argument("--masked_lm_prob", dest="masked_lm_prob", type=float, default=0.15,
                          help="Masked LM probability.")
  arg_parser.add_argument("--short_seq_prob", dest="short_seq_prob", type=float, default=0.1,
                          help="Probability of creating sequences which are shorter than the maximum length.")
  arg_parser.add_argument("--special_tok", dest="special_tok", type=str, default="[EOT]",
                          help="Special Token.")
  args = arg_parser.parse_args()

  create_data = CreateBertPretrainingData(args)

  rng = random.Random(args.random_seed)
  create_data.create_training_instances(
    args.input_file, args.max_seq_length, args.dupe_factor,
    args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq, rng, args.special_tok)

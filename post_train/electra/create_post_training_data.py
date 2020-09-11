# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
import random
import argparse
import pickle

from models.bert import tokenization_bert

class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, input_ids, input_mask, token_type_ids):

    self.input_ids = input_ids
    self.input_mask = input_mask
    self.token_type_ids = token_type_ids

class ExampleBuilder(object):
  """Given a stream of input text, creates pretraining examples."""

  def __init__(self, tokenizer, max_length):
    self._tokenizer = tokenizer
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length

    self.cnt_examples = 0
    self.num_unk_tok = 0

  def add_line(self, line):
    """Adds a line of text to the current example being built."""
    line = line.strip().replace("\n", " ")
    if (not line) and self._current_length != 0:  # empty lines separate docs
      return self._create_example() # one document is finished -> creating examples
    bert_tokens = self._tokenizer.tokenize(line)
    bert_tokens = self._add_special_tokens(bert_tokens, "[EOT]")

    bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
    if 100 in bert_tokids:
      self.num_unk_tok += 1

    self._current_sentences.append(bert_tokids)
    self._current_length += len(bert_tokids)
    if self._current_length >= self._target_length:
      return self._create_example()
    return None

  def _add_special_tokens(self, tokens, special_tok="[EOT]"):
    tokens = tokens + [special_tok]
    return tokens

  def _create_example(self):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      first_segment_target_length = (self._current_length - 3) // 2

    first_segment = []
    second_segment = []
    for sentence in self._current_sentences:
      # the sentence goes to the first segment if
      # (1) the first segment is empty,
      # (2) the sentence doesn't put the first segment over length or
      # (3) 50% of the time when it does put the first segment over length
      if (len(first_segment) == 0 or len(first_segment) + len(sentence) < first_segment_target_length or
          (len(second_segment) == 0 and len(first_segment) < first_segment_target_length and random.random() < 0.5)):
        first_segment += sentence
      else:
        second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:self._max_length - 2]
    second_segment = second_segment[:max(0, self._max_length - len(first_segment) - 3)]

    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0

    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length)
    else:
      self._target_length = self._max_length

    return self._make_example(first_segment, second_segment)

  def _make_example(self, first_segment, second_segment):
    """Converts two "segments" of text into a tf.train.Example."""
    vocab = self._tokenizer.vocab
    input_ids = [vocab["[CLS]"]] + first_segment + [vocab["[SEP]"]]
    segment_ids = [0] * len(input_ids)
    if second_segment:
      input_ids += second_segment + [vocab["[SEP]"]]
      segment_ids += [1] * (len(second_segment) + 1)
    input_mask = [1] * len(input_ids)
    input_ids += [0] * (self._max_length - len(input_ids)) # padding
    input_mask += [0] * (self._max_length - len(input_mask)) # padding
    segment_ids += [0] * (self._max_length - len(segment_ids)) # padding

    self.cnt_examples += 1

    return TrainingInstance(input_ids, input_mask, segment_ids)


class CreateELECTRAPretrainingData(object):
  def __init__(self, args):
    self.args = args
    self._bert_tokenizer_init(args.special_tok, args.bert_pretrained)

  def _bert_tokenizer_init(self, special_tok, bert_pretrained):
    bert_pretrained_dir = os.path.join("./resources", bert_pretrained)
    vocab_file_path = "%s-vocab.txt" % bert_pretrained

    self._tokenizer = tokenization_bert.BertTokenizer(vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path))
    self._tokenizer.add_tokens([special_tok]) # add EOT

    print("BERT tokenizer init completes")

  def create_training_instances(self, input_file, output_file, max_seq_length):
    """Create `TrainingInstance`s from raw text."""
    self._example_builder = ExampleBuilder(self._tokenizer, max_seq_length)
    self._blanks_separate_docs = True

    n_written = 0
    f_writer = open(output_file, "wb")
    """Writes out examples from the provided input file."""
    with open(input_file) as f_reader:
      for line in f_reader:
        line = line.strip()
        if line or self._blanks_separate_docs:
          example = self._example_builder.add_line(line)
          if example:
            if n_written % 10000 == 0:
              print(n_written, self._tokenizer.convert_ids_to_tokens(example.input_ids))
            pickle.dump(example, f_writer)
            n_written += 1

      example = self._example_builder.add_line("")
      if example:
        pickle.dump(example, f_writer) # for the last example instance
        n_written += 1
    print("Total Number of example created : ", self._example_builder.cnt_examples, ":", self._example_builder.num_unk_tok)
    f_writer.close()

if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(description="Bert / Create Pretraining Data")
  arg_parser.add_argument("--input_file", dest="input_file", type=str, required=True,
                          help="Input raw text file (or comma-separated list of files). "
                               "e.g., ./data/ubuntu_corpus_v1/ubuntu_post_training.txt")
  arg_parser.add_argument("--output_file", dest="output_file", type=str, required=True,
                          help="Output example pkl. e.g., ./data/ubuntu_corpus_v1/ubuntu_electra_post_training.pkl")
  arg_parser.add_argument("--bert_pretrained", dest="bert_pretrained", type=str, required=True,
                          help="bert-base-wwm-chinese")
  arg_parser.add_argument("--max_seq_length", dest="max_seq_length", type=int, default=512,
                          help="Maximum sequence length.")
  arg_parser.add_argument("--special_tok", dest="special_tok", type=str, default="[EOT]",
                          help="Special Token.")
  args = arg_parser.parse_args()

  create_data = CreateELECTRAPretrainingData(args)
  create_data.create_training_instances(args.input_file, args.output_file, args.max_seq_length)
import os
import sys

sys.path.append(os.getcwd())
import pickle
from tqdm import tqdm

from models.bert import tokenization_bert


class InputExamples(object):
  def __init__(self, utterances, response, label, seq_lengths):
    self.utterances = utterances
    self.response = response
    self.label = label

    self.dialog_len = seq_lengths[0]
    self.response_len = seq_lengths[1]


class ECommerceDataUtils(object):
  def __init__(self, txt_path, bert_pretrained_dir, bert_pretrained):
    # bert_tokenizer init
    self.txt_path = txt_path
    self._bert_tokenizer_init(bert_pretrained_dir, bert_pretrained)

  def _bert_tokenizer_init(self, bert_pretrained_dir, bert_pretrained='bert-base-chinese'):

    self._bert_tokenizer = tokenization_bert.BertTokenizer(
      vocab_file=os.path.join(os.path.join(bert_pretrained_dir, bert_pretrained),
                              "%s-vocab.txt" % bert_pretrained))
    print("BERT tokenizer init completes")

  def read_raw_file(self, data_type):
    print("Loading raw txt file...")

    ubuntu_path = self.txt_path % data_type  # train, dev, test
    with open(ubuntu_path, "r", encoding="utf8") as fr_handle:
      data = [line.strip() for line in fr_handle if len(line.strip()) > 0]
      print("(%s) total number of sentence : %d" % (data_type, len(data)))

    return data

  def make_examples_pkl(self, data, pkl_path):

    with open(pkl_path, "wb") as pkl_handle:
      for dialog in tqdm(data):
        dialog_data = dialog.split("\t")
        label = dialog_data[0]
        utterances = []
        dialog_len = []

        for utt in dialog_data[1:-1]:
          utt_tok = self._bert_tokenizer.tokenize(utt)
          utterances.append(utt_tok)
          dialog_len.append(len(utt_tok))
        response = self._bert_tokenizer.tokenize(dialog_data[-1])

        pickle.dump(InputExamples(
          utterances=utterances, response=response, label=int(label),
          seq_lengths=(dialog_len, len(response))), pkl_handle)

    print(pkl_path, " save completes!")


if __name__ == '__main__':

  douban_raw_path = "./data/douban/%s.txt"
  douban_pkl_path = "./data/douban/douban_%s.pkl"
  bert_pretrained_dir = "./resources"

  ecommerce_utils = ECommerceDataUtils(douban_raw_path, bert_pretrained_dir, "bert-base-chinese")

  # response seleciton fine-tuning pkl creation
  for data_type in ["test", "train"]:
    data = ecommerce_utils.read_raw_file(data_type)
    ecommerce_utils.make_examples_pkl(data, douban_pkl_path % data_type)

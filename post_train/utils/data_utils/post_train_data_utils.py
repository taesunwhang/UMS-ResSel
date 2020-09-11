import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm

from models.bert import tokenization_bert

class InputExamples(object):
  def __init__(self, utterances, response, label, seq_lengths):

    self.utterances = utterances
    self.response = response
    self.label = label

    self.dialog_len = seq_lengths[0]
    self.response_len = seq_lengths[1]

class PostTrainDataUtils(object):
  def __init__(self, txt_path, bert_pretrained_dir, bert_pretrained):
    # bert_tokenizer init
    self.txt_path = txt_path
    self._bert_tokenizer_init(bert_pretrained_dir, bert_pretrained)

  def _bert_tokenizer_init(self, bert_pretrained_dir, bert_pretrained):

    self._bert_tokenizer = tokenization_bert.BertTokenizer(
      vocab_file=os.path.join(os.path.join(bert_pretrained_dir, bert_pretrained), "%s-vocab.txt" % bert_pretrained))
    print("BERT tokenizer init completes")

  def read_raw_file(self, ):
    print("Loading raw txt file...")

    with open(self.txt_path, "r", encoding="utf8") as fr_handle:
      data = [line.strip() for line in fr_handle if len(line.strip()) > 0]
      print("(train) total number of sentence : %d" % (len(data)))

    return data

  def make_post_training_corpus(self, data, post_training_path):
    with open(post_training_path, "w", encoding="utf-8") as fw_handle:
      cnt = 0
      for document in tqdm(data):
        dialog_data = document.split("\t")
        if dialog_data[0] == '0':
          continue
        for utt in dialog_data[1:-1]:
          if len(utt) == 0:
            continue
          fw_handle.write(utt.strip() + "\n")
        fw_handle.write("\n")
        cnt+=1

if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Pretraining Corpus Creation")
  arg_parser.add_argument("--raw_path", dest="raw_path", type=str, required=True,
                          help="./data/ubuntu_corpus_v1/%s.txt")
  arg_parser.add_argument("--output_path", dest="output_path", type=str, required=True,
                          help="./data/ubuntu_corpus_v1/ubuntu_post_training.txt")
  arg_parser.add_argument("--bert_pretrained", dest="bert_pretrained", type=str,
                          required=True, help="bert-base-wwm-chinese")
  args = arg_parser.parse_args()

  #raw_path = "./data/douban/train.txt"
  #output_path = "./data/ubuntu_corpus_v1/ubuntu_post_training.txt"
  #bert_pretrained = "bert-base-wwm-chinese"

  data_utils = PostTrainDataUtils(args.raw_path, "./resources", args.bert_pretrained)

  # domain post training corpus creation
  data = data_utils.read_raw_file()
  data_utils.make_post_training_corpus(data, args.output_path)


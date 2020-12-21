from models.bert_cls import BertCls
from models.bert_eot import BertEOT
from models.bert_insertion import BertInsertion


def Model(hparams, *args):
  name_model_map = {
    "bert_base": BertCls,
    "bert_post": BertCls,

    "electra_base": BertCls,
    "electra_post": BertCls,

    "bert_base_eot": BertEOT,
    "bert_post_eot": BertEOT,
  }

  return name_model_map[hparams.model_type](hparams, *args)

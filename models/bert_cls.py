import os
import torch.nn as nn

from models.bert_insertion import BertInsertion
from models.bert_deletion import BertDeletion
from models.bert_search import BertSearch

class BertCls(nn.Module):
  def __init__(self, hparams):
    super(BertCls, self).__init__()
    self.hparams = hparams

    pretraomed_config = hparams.pretrained_config.from_pretrained(
      os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                   "%s-config.json" % self.hparams.bert_pretrained),
    )
    self._model = hparams.pretrained_model.from_pretrained(
      os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained, self.hparams.bert_checkpoint_path),
      config=pretraomed_config
    )

    num_new_tok = 0
    if self.hparams.model_type.startswith("bert_base") or self.hparams.model_type.startswith("electra_base"):
      if self.hparams.do_eot:
        num_new_tok += 1
      # bert_post already has [EOT]

    if self.hparams.do_sent_insertion:
      num_new_tok += 1 # [INS]
    if self.hparams.do_sent_deletion:
      num_new_tok += 1  # [INS]
    if self.hparams.do_sent_search:
      num_new_tok += 1 # [SRCH]

    self._model.resize_token_embeddings(self._model.config.vocab_size + num_new_tok)  # [EOT]

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.bert_hidden_dim, 1)
    )

    if self.hparams.do_sent_insertion:
      self._bert_insertion = BertInsertion(hparams, self._model)
    if self.hparams.do_sent_deletion:
      self._bert_deletion = BertDeletion(hparams, self._model)
    if self.hparams.do_sent_search:
      self._bert_search = BertSearch(hparams, self._model)

    self._criterion = nn.BCEWithLogitsLoss()

  def forward(self, batch):
    logits, res_sel_loss, ins_loss, del_loss, srch_loss = None, None, None, None, None

    if self.hparams.do_response_selection:
      outputs = self._model(
        batch["res_sel"]["anno_sent"],
        token_type_ids=batch["res_sel"]["segment_ids"],
        attention_mask=batch["res_sel"]["attention_mask"]
      )
      bert_outputs = outputs[0]
      cls_logits = bert_outputs[:,0,:] # bs, bert_output_size
      logits = self._classification(cls_logits) # bs, 1
      logits = logits.squeeze(-1)
      res_sel_loss = self._criterion(logits, batch["res_sel"]["label"])

    if self.hparams.do_sent_insertion and (self.training or self.hparams.pca_visualization):
      ins_loss = self._bert_insertion(batch["ins"], batch["res_sel"]["label"])
    if self.hparams.do_sent_deletion and (self.training or self.hparams.pca_visualization):
      del_loss = self._bert_deletion(batch["del"], batch["res_sel"]["label"])
    if self.hparams.do_sent_search and (self.training or self.hparams.pca_visualization):
      srch_loss = self._bert_search(batch["srch"], batch["res_sel"]["label"])

    return logits, (res_sel_loss, ins_loss, del_loss, srch_loss)
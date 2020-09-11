import os
import torch
import torch.nn as nn

from models.bert_insertion import BertInsertion
from models.bert import modeling_bert, configuration_bert

class BertEOT(nn.Module):
  def __init__(self, hparams):
    super(BertEOT, self).__init__()
    self.hparams = hparams
    bert_config = configuration_bert.BertConfig.from_pretrained(
      os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                   "%s-config.json" % self.hparams.bert_pretrained),
    )
    self._bert_model = modeling_bert.BertModel.from_pretrained(
      os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained, self.hparams.bert_checkpoint_path),
      config=bert_config
    )

    if self.hparams.do_eot and self.hparams.model_type.startswith("bert_base"):
      self._bert_model.resize_token_embeddings(self._bert_model.config.vocab_size + 2)  # [EOT]

    self._classification = nn.Sequential(
        nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
        nn.Linear(self.hparams.bert_hidden_dim * 2, 1)
      )
    if self.hparams.do_sent_insertion:
      self._bert_insertion = BertInsertion(hparams, self._bert_model)

    self._criterion = nn.BCEWithLogitsLoss()

  def forward(self, batch):
    bert_outputs, _ = self._bert_model(
      batch["anno_sent"],
      token_type_ids=batch["segment_ids"],
      attention_mask=batch["attention_mask"]
    )
    # bert_outputs : [bs, seq_len, bert_output_size]

    # eot : 1
    # eot : 0

    eot_feats = []
    for batch_idx, eot_pos in enumerate(batch["eot_pos"]):
      eot_pos_nonzero = eot_pos.nonzero().view(-1)

      dialog_eot_outputs = bert_outputs[batch_idx,eot_pos_nonzero[:-1],:]
      dialog_pooled_outputs = torch.max(dialog_eot_outputs, dim=0)[0]

      response_eot_outputs = bert_outputs[batch_idx,eot_pos_nonzero[-1],:]

      # bert_output_size * 3
      eot_feats.append(torch.cat((dialog_pooled_outputs, response_eot_outputs), dim=-1))

    eot_logits = torch.stack(eot_feats, dim=0) # bs, bert_output_size * 2

    logits = self._classification(eot_logits) # bs, 1
    logits = logits.squeeze(-1)

    insertion_loss = None
    if self.hparams.do_sent_insertion:
      insertion_loss = self._bert_insertion(batch)
    eot_loss = self._criterion(logits, batch["label"])

    return logits, (eot_loss, insertion_loss)
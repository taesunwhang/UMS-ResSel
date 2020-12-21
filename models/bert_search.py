import torch
import torch.nn as nn
import pickle


class BertSearch(nn.Module):
  def __init__(self, hparams, pretrained_model):
    super(BertSearch, self).__init__()
    self.hparams = hparams
    self._model = pretrained_model

    self.hparams = hparams
    self._model = pretrained_model

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.bert_hidden_dim, 1)
    )

    if self.hparams.auxiliary_loss_type == "softmax":
      self._criterion = nn.CrossEntropyLoss()
    elif self.hparams.auxiliary_loss_type == "sigmoid":
      self._criterion = nn.BCEWithLogitsLoss()
    else:
      raise NotImplementedError

  def forward(self, batch, batch_ressel_label):
    outputs = self._model(
      batch["anno_sent"],
      token_type_ids=batch["segment_ids"],
      attention_mask=batch["attention_mask"]
    )
    bert_outputs = outputs[0]

    srch_losses = []

    for batch_idx, ins_pos in enumerate(batch["srch_pos"]):
      if batch["label"][batch_idx] == -1:
        continue

      if batch_ressel_label[batch_idx] == 0:
        continue

      srch_pos_nonzero = ins_pos.nonzero().view(-1)

      dialog_srch_out = bert_outputs[batch_idx, srch_pos_nonzero, :]  # num_utterances, 768
      srch_logits = self._classification(dialog_srch_out)  # num_utterances, 1
      srch_logits = srch_logits.squeeze(-1)  # num_utterances

      target_id = batch["label"][batch_idx]

      if self.hparams.auxiliary_loss_type == "softmax":
        srch_loss = self._criterion(srch_logits.unsqueeze(0), target_id.unsqueeze(0))
      elif self.hparams.auxiliary_loss_type == "sigmoid":
        srch_label = torch.eye(srch_pos_nonzero.size(0))[target_id].to(torch.cuda.current_device())
        srch_loss = self._criterion(srch_logits, srch_label)
      else:
        raise NotImplementedError

      srch_losses.append(srch_loss)
    if len(srch_losses) == 0:
      search_loss = torch.tensor(0).float().to(torch.cuda.current_device())
    else:
      search_loss = torch.mean(torch.stack(srch_losses, dim=0), dim=-1)

    return search_loss

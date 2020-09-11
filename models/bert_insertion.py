import torch
import torch.nn as nn
import pickle

class BertInsertion(nn.Module):
  def __init__(self, hparams, pretrained_model):
    super(BertInsertion, self).__init__()
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

    if self.hparams.pca_visualization:
      pca_handle = open("/data/taesunwhang/response_selection/visualization/%s/ins_token_representation.pkl"
                        % self.hparams.task_name, "ab")
      print(pca_handle)

    ins_losses = []
    for batch_idx, ins_pos in enumerate(batch["ins_pos"]):
      if batch["label"][batch_idx] == -1:
        continue

      if batch_ressel_label[batch_idx] == 0:
        continue

      ins_pos_nonzero = ins_pos.nonzero().view(-1)

      dialog_ins_out = bert_outputs[batch_idx, ins_pos_nonzero, :] # num_utterances, 768
      ins_logits = self._classification(dialog_ins_out)  # num_utterances, 1
      ins_logits = ins_logits.squeeze(-1) # num_utterances

      target_id = batch["label"][batch_idx]

      if self.hparams.pca_visualization:
        pickle.dump([dialog_ins_out.to("cpu").tolist(), target_id.to("cpu").tolist()], pca_handle)

      if self.hparams.auxiliary_loss_type == "softmax":
        ins_loss = self._criterion(ins_logits.unsqueeze(0), target_id.unsqueeze(0))
      elif self.hparams.auxiliary_loss_type == "sigmoid":
        ins_label = torch.eye(ins_pos_nonzero.size(0))[target_id].to(torch.cuda.current_device())
        ins_loss = self._criterion(ins_logits, ins_label)
      else:
        raise NotImplementedError

      ins_losses.append(ins_loss)

    if len(ins_losses) == 0:
      insertion_loss = torch.tensor(0).float().to(torch.cuda.current_device())
    else:
      insertion_loss = torch.mean(torch.stack(ins_losses, dim=0), dim=-1)

    return insertion_loss
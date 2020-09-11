import torch
import torch.nn as nn
import pickle

class BertDeletion(nn.Module):
  def __init__(self, hparams, pretrained_model):
    super(BertDeletion, self).__init__()
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
      pca_handle = open("/data/taesunwhang/response_selection/visualization/%s/del_token_representation.pkl"
                        % self.hparams.task_name, "ab")
      print(pca_handle)

    del_losses = []
    for batch_idx, del_pos in enumerate(batch["del_pos"]):
      if batch["label"][batch_idx] == -1:
        continue
      if batch_ressel_label[batch_idx] == 0:
        continue

      del_pos_nonzero = del_pos.nonzero().view(-1)
      dialog_del_out = bert_outputs[batch_idx, del_pos_nonzero, :] # num_utterances, 768
      del_logits = self._classification(dialog_del_out)  # num_utterances, 1
      del_logits = del_logits.squeeze(-1) # num_utterances
      target_id = batch["label"][batch_idx]

      if self.hparams.pca_visualization:
        pickle.dump([dialog_del_out.to("cpu").tolist(), target_id.to("cpu").tolist()], pca_handle)

      if self.hparams.auxiliary_loss_type == "softmax":
        del_loss = self._criterion(del_logits.unsqueeze(0), target_id.unsqueeze(0))
      elif self.hparams.auxiliary_loss_type == "sigmoid":
        del_label = torch.eye(del_pos_nonzero.size(0))[target_id].to(torch.cuda.current_device())
        del_loss = self._criterion(del_logits, del_label)
      else:
        raise NotImplementedError

      del_losses.append(del_loss)

    if len(del_losses) == 0:
      deletion_loss = torch.tensor(0).float().to(torch.cuda.current_device())
    else:
      deletion_loss = torch.mean(torch.stack(del_losses, dim=0), dim=-1)

    return deletion_loss
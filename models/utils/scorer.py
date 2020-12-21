import numpy as np


def calculate_candidates_ranking(prediction, ground_truth, eval_candidates_num=10):
  total_num_split = len(ground_truth) / eval_candidates_num

  pred_split = np.split(prediction, total_num_split)
  gt_split = np.split(np.array(ground_truth), total_num_split)
  orig_rank_split = np.split(np.tile(np.arange(0, eval_candidates_num), int(total_num_split)), total_num_split)
  stack_scores = np.stack((gt_split, pred_split, orig_rank_split), axis=-1)

  rank_by_pred_l = []
  for i, stack_score in enumerate(stack_scores):
    rank_by_pred = sorted(stack_score, key=lambda x: x[1], reverse=True)
    rank_by_pred = np.stack(rank_by_pred, axis=-1)
    rank_by_pred_l.append(rank_by_pred[0])
  rank_by_pred = np.array(rank_by_pred_l)

  pos_index = []
  for sorted_score in rank_by_pred:
    curr_cand = []
    for p_i, score in enumerate(sorted_score):
      if int(score) == 1:
        curr_cand.append(p_i)
    pos_index.append(curr_cand)

  return rank_by_pred, pos_index, stack_scores


def logits_recall_at_k(pos_index, k_list=[1, 2, 5, 10]):
  # 1 dialog, 10 response candidates ground truth 1 or 0
  # prediction_score : [batch_size]
  # target : [batch_size] e.g. 1 0 0 0 0 0 0 0 0 0
  # e.g. batch : 100 -> 100/10 = 10

  num_correct = np.zeros([len(pos_index), len(k_list)])
  index_dict = dict()
  for i, p_i in enumerate(pos_index):
    index_dict[i] = p_i

  # case for douban : more than one correct answer case
  for i, p_i in enumerate(pos_index):
    if len(p_i) == 1 and p_i[0] >= 0:
      for j, k in enumerate(k_list):
        if p_i[0] + 1 <= k:
          num_correct[i][j] += 1
    elif len(p_i) > 1:
      for j, k in enumerate(k_list):
        all_recall_at_k = []
        for cand_p_i in p_i:
          if cand_p_i + 1 <= k:
            all_recall_at_k.append(1)
          else:
            all_recall_at_k.append(0)
        num_correct[i][j] += np.mean(all_recall_at_k)

  return np.sum(num_correct, axis=0)


def logits_mrr(pos_index):
  mrr = []
  for i, p_i in enumerate(pos_index):
    if len(p_i) > 0 and p_i[0] >= 0:
      mrr.append(1 / (p_i[0] + 1))
    elif len(p_i) == 0:
      mrr.append(0)  # no answer

  return np.sum(mrr)


def precision_at_one(rank_by_pred):
  num_correct = [0] * rank_by_pred.shape[0]
  for i, sorted_score in enumerate(rank_by_pred):
    for p_i, score in enumerate(sorted_score):
      if p_i == 0 and int(score) == 1:
        num_correct[i] = 1
        break

  return np.sum(num_correct)


def mean_average_precision(pos_index):
  map = []
  for i, p_i in enumerate(pos_index):
    if len(p_i) > 0:
      all_precision = []
      for j, cand_p_i in enumerate(p_i):
        all_precision.append((j + 1) / (cand_p_i + 1))
      curr_map = np.mean(all_precision)
      map.append(curr_map)
    elif len(p_i) == 0:
      map.append(0)  # no answer

  return np.sum(map)

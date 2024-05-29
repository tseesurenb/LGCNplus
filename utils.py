'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''

from collections import defaultdict
import numpy as np
import sys

def get_recall_at_k(input_edge_index,
                    input_edge_values,
                    pred_ratings,
                    k=10,
                    threshold=3.5):
    user_item_rating_list = defaultdict(list)
    
    for i in range(len(input_edge_index[0])):
        src = input_edge_index[0][i].item()
        dest = input_edge_index[1][i].item()
        true_rating = input_edge_values[i].item()
        pred_rating = pred_ratings[i].item()
        
        user_item_rating_list[src].append((pred_rating, true_rating))
        
    recalls = dict()
    precisions = dict()

    for user_id, user_ratings in user_item_rating_list.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((pred_r >= threshold) for (pred_r, _) in user_ratings[:k])
        
        n_rel_and_rec_k = sum(((true_r >= threshold) and (pred_r >= threshold)) \
                                for (pred_r, true_r) in user_ratings[:k])
        
        precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        
    overall_recall = sum(rec for rec in recalls.values()) / len(recalls)
    overall_precision = sum(prec for prec in precisions.values()) / len(precisions)
    
    #print("Calculating recall and precision...")
    #print(f"Threshold: {threshold}", "Top-K: ", k, "len of recalls: ", len(recalls), "len of precisions: ", len(precisions))
    #print("Overall Recall: ", overall_recall, "Overall Precision: ", overall_precision)
    
    return overall_recall, overall_precision

def print_rmse(ITERATIONS, iter, train_loss, val_loss, recall, precision, time):
    
    f_train_loss = "{:.3f}".format(round(np.sqrt(train_loss.item()), 3))
    f_val_loss = "{:.3f}".format(round(np.sqrt(val_loss.item()), 3))
    f_recall = "{:.3f}".format(round(recall, 3))
    f_precision = "{:.3f}".format(round(precision, 3))
    f_time = "{:.2f}".format(round(time, 2))
    f_iter = "{:.0f}".format(iter)
    
    sys.stdout.write(f"\rEpoch {f_iter}/{ITERATIONS} - Train Loss: {train_loss:.3f}, "
                     f"Val Loss: {val_loss:.3f}, Recall: {f_recall}, Precision: {f_precision}, Time: {f_time} s")
    sys.stdout.flush()
    
    #print(f"[Epoch ({f_time}) {f_iter}]\tRMSE(train->val): {f_train_loss}"
    #      f"\t-> {f_val_loss} | "
    #      f"Recall, Prec:{f_recall, f_precision}")
  

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 32)
    
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

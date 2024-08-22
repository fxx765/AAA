import dill
import numpy as np
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score, auc
import pickle
import gzip
import torch
from torch_geometric.data import HeteroData
import torch_geometric.utils as tgu
from shutil import which
import torch
import numpy as np
import copy
import random

# Define a function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)

# Define a function to load all required data
def data_loader(load_graph=False):
    if load_graph:
        records_path = "./data/output_kg/records_graph_virtualnode.pkl.gz"
    else:
        records_path = "./data/output_kg/records_final_list.pkl.gz"
    ehr_adj_path = "./data/output_kg/ehr_adj_final.pkl"
    ddi_adj_path = "./data/output_kg/ddi_A_final.pkl"
    ddi_mask_path = "./data/output_kg/ddi_mask_H.pkl"
    atc3toSMILES_path = "./data/output_kg/atc3toSMILES.pkl"
    voc_path = "./data/output_kg/voc_final.pkl"

    records = pickle.load(gzip.open(records_path, 'r' ))
    ehr_adj = dill.load(open(ehr_adj_path, "rb"))
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask = dill.load(open(ddi_mask_path, "rb"))
    atc3toSMILES = dill.load(open(atc3toSMILES_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))

    return records, ehr_adj, ddi_adj, ddi_mask, atc3toSMILES, voc

# Define a function to split the data
def data_splitter(records, args):
    split_point = int(len(records) * args.portion_train)
    data_train = records[:split_point]
    eval_len = int(len(records[split_point:]) / 2)
    data_test = records[split_point:split_point+eval_len]
    data_valid = records[split_point+eval_len:]

    return data_train, data_valid, data_test

# Define a function to augment the graph
def augment_graph(x, edge_index, edge_type, n_edge_types=0, p=0.1):
    r = np.random.choice(np.arange(2))
    if r == 1:
        edge_index, edge_mask = tgu.dropout_edge(edge_index, p=p)
        edge_type = edge_type[edge_mask]
    return x, edge_index, edge_type

# Define a series of functions to calculate performance metrics
def calculate_ddi_rate_score(record, ddi_adj):
    # ddi rate
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt

def calculate_ml_ddi_rate_score(y_pred, ddi_adj):
    all_cnt = 0
    dd_cnt = 0
    med_cnt = 0
    visit_cnt = 0
    for adm in y_pred:
        med_code_set = np.where(adm == 1)[0]
        visit_cnt += 1
        med_cnt += len(med_code_set)
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
                    dd_cnt += 1
    ddi_rate = dd_cnt / all_cnt
    meds = med_cnt / visit_cnt
    return ddi_rate, meds

# Define function to measure the evaluation metric for classification
def calculate_performance(y_gts, y_hat_preds, y_hat_probs):
    def calculate_average_jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def calculate_precision(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def calculate_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def calculate_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def calculate_average_precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    metric_jaccard = calculate_average_jaccard(y_gts, y_hat_preds)
    metric_prauc = calculate_average_precision_auc(y_gts, y_hat_probs)
    metric_prec = calculate_precision(y_gts, y_hat_preds)
    metric_recall = calculate_recall(y_gts, y_hat_preds)
    metric_f1 = calculate_f1(metric_prec, metric_recall)
    return metric_jaccard, metric_prauc, np.mean(metric_prec), np.mean(metric_recall), np.mean(metric_f1)
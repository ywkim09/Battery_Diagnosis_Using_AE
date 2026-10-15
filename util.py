import torch

from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'weights'
INPUT_DIM = 180
NUMERIC_DIM = 29
CHUNK_SIZE = 50000
NUM_WORKERS = 4

LayerOutput = namedtuple(typename='LayerOutput',
                         field_names=['dist', 'input', 'mu', 'sigma', 'sample', 'sigmoids'])

def get_node_key(node):
    '''
    Get the dictionary key for a node.

    Parameters:
    node (torch.tensor): node to get key for

    Returns:
    (tuple): dictionary key for given node
    '''
    return tuple(node.tolist())

def compute_thresh(scores, pctg = 0.95):
    '''
    Compute threshold based on given percentile of scores.

    Parameters:
    scores (np.ndarray): scores to compute threshold based on
    pctg (float): percentile for threshold, default 0.95

    Returns:
    (float): computed threshold
    '''
    scores = np.sort(scores)
    idx = int(len(scores)*pctg)
    thresh = scores[idx]
    return thresh

def performance(labels, scores, thresh = None):
    '''
    Compute classification performance metrics based on given scores and truth labels.

    Parameters:
    labels (np.ndarray): ground truth labels
    scores (np.ndarray): predicted scores 
    thresh (float): threshold for threshold-independent metrics, default None

    Returns:
    (float): computed AUROC
    (float): computed AUPRC
    (float): computed F1 score
    (float): computed macro F1 score
    (float): computed accuracy
    '''
    scores = np.nan_to_num(scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr) if labels.sum() > 0 else None
    f1, mf1, acc = None, None, None
    if thresh is not None:
        preds = scores > thresh
        f1 = f1_score(labels, preds)
        mf1 = f1_score(labels, preds, average = 'macro')
        acc = accuracy_score(labels, preds)
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision) if labels.sum() > 0 else None

    return auroc, auprc, f1, mf1, acc

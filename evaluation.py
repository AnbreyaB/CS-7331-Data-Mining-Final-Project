from scipy.stats import chisquare
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    jaccard_score,
    silhouette_score,
    confusion_matrix, 
    roc_auc_score, 
    ConfusionMatrixDisplay)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def evaluation_aggregate(result):
    eval_aggregate = {}

    if (result["y_test"] is not None and result["y_pred"] is not None):
        accuracy = accuracy_score(result["y_test"], result["y_pred"])
        precision = precision_score(result["y_test"], result["y_pred"])
        recall = recall_score(result["y_test"], result["y_pred"])
        f1 = f1_score(result["y_test"], result["y_pred"])

        eval_aggregate["accuracy"] = accuracy
        eval_aggregate["precision"] = precision
        eval_aggregate["recall"] = recall
        eval_aggregate["f1"] = f1

    return eval_aggregate

def calc_support():
    return

def calc_confidence():
    return

def calc_lift():
    return

def calc_accuracy():
    return

def calc_precision():
    return

def calc_recall():
    return

def calc_f1():
    return

def calc_chisquared():
    return

def calc_cosine():
    return

def calc_jaccard():
    return

def calc_silhouette():
    return

def calc_allconf():
    return

def calc_maxconf():
    return


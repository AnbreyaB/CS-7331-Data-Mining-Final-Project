from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    mean_absolute_error,
    jaccard_score,
    silhouette_score)
from pydunn import dunn
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def evaluation_aggregate(result):
    eval_aggregate = {}

    if ("y_test" in result and "y_pred" in result):
        accuracy = accuracy_score(result["y_test"], result["y_pred"])
        precision = precision_score(result["y_test"], result["y_pred"])
        recall = recall_score(result["y_test"], result["y_pred"])
        f1 = f1_score(result["y_test"], result["y_pred"])
        jaccard = jaccard_score(result["y_test"], result["y_pred"])
        mae = mean_absolute_error(result["y_test"], result["y_pred"])

        eval_aggregate["accuracy"] = accuracy
        eval_aggregate["precision"] = precision
        eval_aggregate["recall"] = recall
        eval_aggregate["f1"] = f1
        eval_aggregate["jaccard"] = jaccard
        eval_aggregate["mean absolute error"] = mae

    elif ("X" in result and "labels" in result):
        silhouette = silhouette_score(result["X"], result["labels"])
        km_dunn = dunn(result["labels"], euclidean_distances(result["X"]))

        eval_aggregate["silhouette"] = silhouette
        eval_aggregate["dunn"] = km_dunn

    return eval_aggregate


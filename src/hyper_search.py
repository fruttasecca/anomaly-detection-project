#!/usr/bin/python3
"""
Script to do hyper parameter search.
"""
import sys
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import roc_auc_score, average_precision_score

from Loda import Loda
from SomAnomalyDetector import SomAnomalyDetector
from dataset import Dataset


def loader_generic(file):
    """
    Given a file with 2 columns, 1 for data 1 for value, return a numpy ndarry of values from the second column, keeping
    order.
    :param file: Csv file with 2 columns.
    :return: Numpy ndarray of values in the second column of the file.
    """
    data = pd.read_csv(file)
    return data["value"].values, data["anomaly"].values


def get_thres(labels, scores, beta):
    thres = np.arange(0, scores.max(), 0.01)
    thres = thres[::-1]

    best_thres = 0
    best_f1 = 0
    best_fpr = 0
    best_rfpr_ratio = 0
    best_precision = 0
    best_recall = 0
    pred = 0

    # all zeros at start, set to 2 when the threshold gets low enough
    predicted_anomalies = np.zeros(len(scores))

    # sorted scores, descending
    sorted_score = list(np.sort(scores)[::-1])
    # index of sorted scores, descending
    sorted_idx = list(np.argsort(scores)[::-1])

    # for each thres compute f1
    for th in thres:
        # flag as anomalies idx with scores above th
        while len(sorted_score) > 0 and sorted_score[0] >= th:
            predicted_anomalies[sorted_idx[0]] = 2
            del sorted_idx[0]
            del sorted_score[0]

        # get stats
        idx = predicted_anomalies + labels
        tn = (idx == 0.0).sum().item()  # tn
        fn = (idx == 1.0).sum().item()  # fn
        fp = (idx == 2.0).sum().item()  # fp
        tp = (idx == 3.0).sum().item()  # tp
        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)

        # get F1

        f1 = ((1 + beta * beta) * p * r) / (beta * beta * p + r + 1e-7)

        # FPR
        fpr = fp / (fp + tn + 1e-7)

        # recall / FPR
        rfpr_ratio = r / (fpr + 1e-7)

        if f1 > best_f1:
            best_thres = th
            best_f1 = f1
            best_fpr = fpr
            best_rfpr_ratio = rfpr_ratio
            best_precision = p
            best_recall = r
            pred = predicted_anomalies[:] / 2

    prc = average_precision_score(labels, scores, average=None)
    roc = roc_auc_score(labels, scores, average=None)

    return best_thres, best_f1, best_fpr, best_rfpr_ratio, best_precision, best_recall, np.sum(pred), np.sum(
        labels), np.sum(labels * pred), prc, roc


def search_loda(dataset, iterations, output, multiplier):
    name = dataset
    # saving runs into a dataframe
    columns = ["smoothing", "normalization", "wsize", "memory", "thres", "F1", "beta", "FPR", "RFPR", "Prec", "Rec",
               "tot_pred", "tot_labels", "tot_correctly_pred", "prc", "roc", "order"]
    df_runs = pd.DataFrame(columns=columns)

    # parameters of the search (seed, data, number of trials, etc.)
    data, labels = loader_generic(dataset)

    # parameters of the model
    smoothing = [True, False]
    order = [1, 3, 5, 8, 20, 45]
    normalization = [True, False]
    window_size = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    memory = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    params = [smoothing, normalization, window_size, memory, order]
    params = list(itertools.product(*params))
    params = np.array(params)
    indices = np.random.randint(0, params.shape[0], iterations)
    params = params[indices]

    for ite in range(iterations):
        # get parameters for this run
        tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory, tmp_order = params[ite]
        tmp_window_size = int(tmp_window_size) * multiplier
        tmp_window_size = tmp_window_size + 1 if tmp_window_size % 2 == 0 else tmp_window_size
        if (tmp_smoothing and tmp_window_size < tmp_order + 2) or (name == "riccione" and tmp_smoothing):
            continue

        # init
        dataset = Dataset(data, tmp_smoothing, tmp_normalization, tmp_window_size, 1, tmp_order)
        scores = np.zeros(len(dataset.data))
        detector = Loda(tmp_window_size, tmp_memory)

        # give score to each window
        for i in range(len(dataset)):
            data_point = dataset[i]
            window_score = detector.add_data_point(data_point)
            # update score of elements in window
            for u in range(i, i + tmp_window_size):
                scores[u] = max(scores[u], window_score)

        # check which threshold would give the best f1
        beta = 1
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, prc, roc = get_thres(labels[:len(scores)], scores,
                                                                                        beta)
        df_runs = df_runs.append(pd.DataFrame([[tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory, thres,
                                                f1, beta, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, prc, roc,
                                                tmp_order]],
                                              columns=columns), ignore_index=True)
        beta = 0.1
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, prc, roc = get_thres(labels[:len(scores)], scores,
                                                                                        beta)
        df_runs = df_runs.append(pd.DataFrame([[tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory, thres,
                                                f1, beta, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, prc, roc,
                                                tmp_order]],
                                              columns=columns), ignore_index=True)
        df_runs.to_csv(output)


def search_som(dataset, iterations, output, multiplier):
    # msg to print at the end of each iteration
    name = dataset
    columns = ["smoothing", "normalization", "dimension", "wsize", "sigma", "update_weight", "decay_period", "thres",
               "F1", "beta", "FPR", "RFPR", "Prec", "Rec", "tot_pred", "tot_labels", "tot_correctly_pred", "prc", "roc",
               "order"]
    df_runs = pd.DataFrame(columns=columns)

    # parameters of the search (seed, data, number of trials, etc.)
    data, labels = loader_generic(dataset)

    # parameters of the model
    smoothing = [True, False]
    order = [1, 3, 5, 8, 20, 45]
    normalization = [True, False]
    dimension = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    window_size = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    sigma = [0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5]
    update_weight = [0.001, 0.005, 0.1, 0.2, 0.3, 0.4]
    decay_period = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 5000, 7000]

    params = [smoothing, normalization, dimension, window_size, sigma, update_weight, decay_period, order]
    params = list(itertools.product(*params))
    params = np.array(params)
    indices = np.random.randint(0, params.shape[0], iterations)
    params = params[indices]

    for ite in range(iterations):
        # get parameters for this run
        tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight, tmp_decay_period, tmp_order = \
            params[ite]
        if (tmp_smoothing and tmp_window_size < tmp_order + 2) or (name == "riccione" and tmp_smoothing):
            continue
        tmp_normalization = int(tmp_normalization)
        tmp_window_size = int(tmp_window_size) * multiplier
        tmp_window_size = tmp_window_size + 1 if tmp_window_size % 2 == 0 else tmp_window_size
        tmp_dimension = int(tmp_dimension)
        tmp_decay_period = int(tmp_decay_period)

        # init
        dataset = Dataset(data, tmp_smoothing, tmp_normalization, tmp_window_size, 1, tmp_order)
        scores = np.zeros(len(dataset.data))
        detector = SomAnomalyDetector(tmp_dimension, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight,
                                      tmp_decay_period, decay_factor=0.5)

        # give score to each window
        for i in range(len(dataset)):
            data_point = dataset[i]
            window_score = detector.add_data_point(data_point)
            # update score of elements in window
            for u in range(i, i + tmp_window_size):
                scores[u] = max(scores[u], window_score)

        # check which threshold would give the best f1
        beta = 1
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores,
                                                                                        beta)
        df_runs = df_runs.append(pd.DataFrame(
            [[tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight,
              tmp_decay_period, thres, f1, beta, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc, tmp_order]],
            columns=columns), ignore_index=True)

        beta = 0.1
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores,
                                                                                        beta)
        df_runs = df_runs.append(pd.DataFrame(
            [[tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight,
              tmp_decay_period, thres, f1, beta, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc, tmp_order]],
            columns=columns), ignore_index=True)
        df_runs.to_csv(output)


if __name__ == "__main__":
    datasets = ["taxi", "machine", "artificial", "riccione"]
    algos = ["loda", "som"]

    # args check
    if len(sys.argv) != 5:
        print("usage: ./script algo dataset iterations output")
        exit()
    algo = sys.argv[1]
    assert algo in algos, "Selected algorithm should be among: %s" % " ".join(algos)

    dataset = sys.argv[2]
    assert dataset in datasets, "Selected dataset should be among: %s" % " ".join(datasets)

    assert sys.argv[3].isdigit(), "iterations should be a number"
    iterations = int(sys.argv[3])

    output = sys.argv[4]

    # window multiplier, for multivariate time series, 1 for single variable
    multiplier = 1
    # get dataset
    if dataset == "taxi":
        dataset = "../data/taxi/nyc_taxi_train.csv"
    elif dataset == "machine":
        dataset = "../data/machine_temperature/machine_temperature_system_failure_train.csv"
    elif dataset == "artificial":
        dataset = "../data/artificial/artificial_train.csv"
    elif dataset == "riccione":
        dataset = "../data/riccione/train.csv"
        multiplier = 24

    # run
    if algo == "loda":
        search_loda(dataset, iterations, output, multiplier)
    elif algo == "som":
        search_som(dataset, iterations, output, multiplier)

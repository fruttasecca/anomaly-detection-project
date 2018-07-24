#!/usr/bin/python3
"""
Script to do hyper parameter search.
"""
import sys
import numpy as np
import pandas as pd
import itertools

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


def get_thres(labels, scores, beta=0.1):
    thres = np.arange(0, scores.max(), 0.01)

    best_thres = 0
    best_F1 = 0
    best_FPR = 0
    best_RFPR_ratio = 0
    best_precision = 0
    best_recall = 0
    pred = 0

    # for each thres compute f1
    best_f1 = 0
    for th in thres:
        # get stats
        predicted_anomalies = [2 * (score >= th) for score in scores]
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
        RFPR_ratio = r / (fpr + 1e-7)

        if f1 > best_f1:
            best_thres = th
            best_F1 = f1
            best_FPR = fpr
            best_RFPR_ratio = RFPR_ratio
            best_precision = p
            best_recall = r
            pred = [pred / 2 for pred in predicted_anomalies]

    print("---")
    print(np.sum(pred))
    print(np.sum(labels))
    print(np.sum(labels * pred))
    return best_thres, best_F1, best_FPR, best_RFPR_ratio, best_precision, best_recall, np.sum(pred), np.sum(
        labels), np.sum(
        labels * pred)


def search_loda(dataset, iterations, output, multiplier, beta):
    # saving runs into a dataframe
    columns = ["smoothing", "normalization", "wsize", "memory", "thres", "F1", "FPR", "RFPR", "Prec", "Rec", "tot_pred",
               "tot_labels", "tot_correctly_pred"]
    df_runs = pd.DataFrame(columns=columns)

    # parameters of the search (seed, data, number of trials, etc.)
    data, labels = loader_generic(dataset)

    # parameters of the model
    smoothing = [True, False]
    normalization = [True, False]
    window_size = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    memory = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    params = [smoothing, normalization, window_size, memory]
    params = list(itertools.product(*params))
    params = np.array(params)
    indices = np.random.randint(0, params.shape[0], iterations)
    params = params[indices]

    for ite in range(iterations):
        # get parameters for this run
        tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory = params[ite]
        tmp_window_size = int(tmp_window_size) * multiplier
        tmp_window_size = tmp_window_size + 1 if tmp_window_size % 2 == 0 else tmp_window_size

        # init
        dataset = Dataset(data, tmp_smoothing, tmp_normalization, tmp_window_size, 1, 1)
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
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor = get_thres(labels[:len(scores)], scores, beta)

        print("--------")
        print("smoothing= %s, normalization= %s, wsize = %s, memory = %s thres= %s -> f1 = %s" % (
            tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory, thres, f1))
        df_runs = df_runs.append(pd.DataFrame([[tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory, thres,
                                                f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor]], columns=columns),
                                 ignore_index=True)
        df_runs.to_csv(output)


def search_som(dataset, iterations, output, multiplier, beta):
    # msg to print at the end of each iteration
    msg = "smoothing= %s, normalization= %s, dimension= %s, wsize= %s, sigma= %s, update_weight= %s " \
          "decay_period= %s, thres= %s -> f1 = %s"
    columns = ["smoothing", "normalization", "dimension", "wsize", "sigma", "update_weight", "decay_period", "thres",
               "F1", "FPR", "RFPR", "Prec", "Rec", "tot_pred", "tot_labels", "tot_correctly_pred"]
    df_runs = pd.DataFrame(columns=columns)

    # parameters of the search (seed, data, number of trials, etc.)
    data, labels = loader_generic(dataset)

    # parameters of the model
    smoothing = [True, False]
    normalization = [True, False]
    dimension = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    window_size = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    sigma = [0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5]
    update_weight = [0.001, 0.005, 0.1, 0.2, 0.3, 0.4]
    decay_period = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 5000, 7000]

    params = [smoothing, normalization, dimension, window_size, sigma, update_weight, decay_period]
    params = list(itertools.product(*params))
    params = np.array(params)
    indices = np.random.randint(0, params.shape[0], iterations)
    params = params[indices]

    for ite in range(iterations):
        # get parameters for this run
        tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight, tmp_decay_period = \
            params[ite]
        tmp_normalization = int(tmp_normalization)
        tmp_window_size = int(tmp_window_size) * multiplier
        tmp_window_size = tmp_window_size + 1 if tmp_window_size % 2 == 0 else tmp_window_size
        tmp_dimension = int(tmp_dimension)
        tmp_decay_period = int(tmp_decay_period)

        # init
        dataset = Dataset(data, tmp_smoothing, tmp_normalization, tmp_window_size, 1, 1)
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
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor = get_thres(labels[:len(scores)], scores, beta)

        print("--------")
        print(msg % (tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight,
                     tmp_decay_period, thres, f1))
        df_runs = df_runs.append(pd.DataFrame(
            [[tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight,
              tmp_decay_period, thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor]], columns=columns),
            ignore_index=True)
        df_runs.to_csv(output)


if __name__ == "__main__":
    datasets = ["taxi", "machine", "artificial", "riccione"]
    algos = ["loda", "som"]

    # args check
    if len(sys.argv) != 6:
        print("usage: ./script algo dataset iterations output beta")
        exit()
    algo = sys.argv[1]
    assert algo in algos, "Selected algorithm should be among: %s" % " ".join(algos)

    dataset = sys.argv[2]
    assert dataset in datasets, "Selected dataset should be among: %s" % " ".join(datasets)

    assert sys.argv[3].isdigit(), "iterations should be a number"
    iterations = int(sys.argv[3])

    output = sys.argv[4]

    beta = float(sys.argv[5])
    print(beta)

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
        search_loda(dataset, iterations, output, multiplier, beta)
    elif algo == "som":
        search_som(dataset, iterations, output, multiplier, beta)

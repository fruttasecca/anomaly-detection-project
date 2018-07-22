#!/usr/bin/python3
"""
Script to do hyper parameter search.
"""

import sys
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import precision_recall_fscore_support

from Loda import Loda
from SomAnomalyDetector import SomAnomalyDetector
from dataset import Dataset
from loaders import loader_generic


def get_thres(labels, scores):
    thres = np.arange(0, 1, 0.01)
    best_f1 = 0
    best_thres = 0
    # for each thres compute f1
    for th in thres:
        predicted_anomalies = [score >= th for score in scores]
        precision, recall, f1, support = precision_recall_fscore_support(labels, predicted_anomalies, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thres = th
    return best_thres, best_f1


def search_loda(dataset, iterations, output):
    # saving runs into a dataframe
    columns = ["smoothing", "normalization", "wsize", "memory", "thres", "f1"]
    df_runs = pd.DataFrame(columns=columns)

    # parameters of the search (seed, data, number of trials, etc.)
    data, labels = loader_generic(dataset)

    # parameters of the model
    smoothing = [True, False]
    normalization = [True, False]
    window_size = [11, 25, 51, 75, 101, 125, 151, 175, 201, 225, 251, 275, 301]
    memory = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    params = [smoothing, normalization, window_size, memory]
    params = list(itertools.product(*params))
    params = np.array(params)
    indices = np.random.randint(0, params.shape[0], iterations)
    params = params[indices]

    for ite in range(iterations):
        # get parameters for this run
        tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory = params[ite]

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
        thres, f1 = get_thres(labels[:len(scores)], scores)

        print("--------")
        print("smoothing= %s, normalization= %s, wsize = %s, memory = %s thres= %s -> f1 = %s" % (
            tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory, thres, f1))
        df_runs = df_runs.append(
            pd.DataFrame([[tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory, thres, f1]], columns=columns),
            ignore_index=True)
        df_runs.to_csv(output)


def search_som(dataset, iterations, output):
    # msg to print at the end of each iteration
    msg = "smoothing= %s, normalization= %s, dimension= %s, wsize= %s, sigma= %s, update_weight= %s " \
          "decay_period= %s, thres= %s -> f1 = %s"
    columns = ["smoothing", "normalization", "dimension", "wsize", "sigma", "update_weight", "decay_period", "thres",
               "f1"]
    df_runs = pd.DataFrame(columns=columns)

    # parameters of the search (seed, data, number of trials, etc.)
    data, labels = loader_generic(dataset)

    # parameters of the model
    smoothing = [True, False]
    normalization = [True, False]
    dimension = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    window_size = [11, 25, 51, 75, 101, 125, 151, 175, 201, 225, 251, 275, 301]
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
        tmp_window_size = int(tmp_window_size)
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
        thres, f1 = get_thres(labels[:len(scores)], scores)
        print("--------")
        print(msg % (tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight,
                     tmp_decay_period, thres, f1))
        df_runs = df_runs.append(pd.DataFrame(
            [[tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight,
              tmp_decay_period, thres, f1]], columns=columns), ignore_index=True)
        df_runs.to_csv(output)


if __name__ == "__main__":
    datasets = ["taxi", "machine"]
    algos = ["loda", "som"]
    np.random.seed(1337)

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

    # get dataset
    if dataset == "taxi":
        dataset = "../data/taxi/nyc_taxi_train.csv"
    elif dataset == "machine":
        dataset = "../data/machine_temperature/machine_temperature_system_failure_train.csv"

    # run
    if algo == "loda":
        search_loda(dataset, iterations, output)
    elif algo == "som":
        search_som(dataset, iterations, output)

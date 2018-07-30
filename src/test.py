#!/usr/bin/python3
"""
Script to test algorihms on datasets.
"""
import sys

import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import roc_auc_score, average_precision_score

from Loda import Loda
from SomAnomalyDetector import SomAnomalyDetector
from dataset import Dataset
import luminol
import luminol.anomaly_detector
from hyper_search import loader_generic, get_thres


def loader_generic(file):
    """
    Given a file with 2 columns, 1 for data 1 for value, return a numpy ndarry of values from the second column, keeping
    order.
    :param file: Csv file with 2 columns.
    :return: Numpy ndarray of values in the second column of the file.
    """
    data = pd.read_csv(file)
    return data["value"].values, data["anomaly"].values


def som_articificial(other=False):
    train = "../data/artificial/artificial_train.csv"
    test = "../data/artificial/artificial_test1.csv"
    if other:
        test = "../data/artificial/artificial_test2.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    wsize = 301
    train = Dataset(train, 1, 1, wsize, 1, 1)
    test = Dataset(test, 1, 1, wsize, 1, 1)

    detector = SomAnomalyDetector(10, 10, wsize, 1, 0.2, 600, decay_factor=0.5)

    # use train set just to get statistics in the model
    for i in range(len(train)):
        detector.add_data_point(train[i])

    # pass on test set
    scores = np.zeros(len(test.data))
    for i in range(len(test)):
        window_score = detector.add_data_point(test[i])
        # update score of elements in window
        for u in range(i, i + wsize):
            scores[u] = max(scores[u], window_score)

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def som_taxi():
    train = "../data/taxi/nyc_taxi_train.csv"
    test = "../data/taxi/nyc_taxi_test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    wsize = 175
    train = Dataset(train, 1, 0, wsize, 1, 1)
    test = Dataset(test, 1, 0, wsize, 1, 1)

    detector = SomAnomalyDetector(8, 8, wsize, 5, 0.001, 1400, decay_factor=0.5)

    # use train set just to get statistics in the model
    for i in range(len(train)):
        detector.add_data_point(train[i])

    # pass on test set
    scores = np.zeros(len(test.data))
    for i in range(len(test)):
        window_score = detector.add_data_point(test[i])
        # update score of elements in window
        for u in range(i, i + wsize):
            scores[u] = max(scores[u], window_score)

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def som_machine():
    train = "../data/machine_temperature/machine_temperature_system_failure_train.csv"
    test = "../data/machine_temperature/machine_temperature_system_failure_test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    wsize = 275
    train = Dataset(train, 1, 0, wsize, 1, 1)
    test = Dataset(test, 1, 0, wsize, 1, 1)

    detector = SomAnomalyDetector(2, 2, wsize, 3, 0.001, 400, decay_factor=0.5)

    # use train set just to get statistics in the model
    for i in range(len(train)):
        detector.add_data_point(train[i])

    # pass on test set
    scores = np.zeros(len(test.data))
    for i in range(len(test)):
        window_score = detector.add_data_point(test[i])
        # update score of elements in window
        for u in range(i, i + wsize):
            scores[u] = max(scores[u], window_score)

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def som_riccione():
    train = "../data/riccione/train.csv"
    test = "../data/riccione/test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    wsize = 601
    train = Dataset(train, 0, 0, wsize, 1, 1)
    test = Dataset(test, 0, 0, wsize, 1, 1)

    detector = SomAnomalyDetector(10, 10, wsize, 1.5, 0.1, 1000, decay_factor=0.5)

    # use train set just to get statistics in the model
    for i in range(len(train)):
        detector.add_data_point(train[i])

    # pass on test set
    scores = np.zeros(len(test.data))
    for i in range(len(test)):
        window_score = detector.add_data_point(test[i])
        # update score of elements in window
        for u in range(i, i + wsize):
            scores[u] = max(scores[u], window_score)

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def loda_artificial(other=False):
    train = "../data/artificial/artificial_train.csv"
    test = "../data/artificial/artificial_test1.csv"
    if other:
        test = "../data/artificial/artificial_test2.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    wsize = 225
    train = Dataset(train, 0, 0, wsize, 1, 1)
    test = Dataset(test, 0, 0, wsize, 1, 1)

    detector = Loda(wsize, 1400)

    # use train set just to get statistics in the model
    for i in range(len(train)):
        detector.add_data_point(train[i])

    # pass on test set
    scores = np.zeros(len(test.data))
    for i in range(len(test)):
        window_score = detector.add_data_point(test[i])
        # update score of elements in window
        for u in range(i, i + wsize):
            scores[u] = max(scores[u], window_score)

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def loda_taxi():
    train = "../data/taxi/nyc_taxi_train.csv"
    test = "../data/taxi/nyc_taxi_test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    wsize = 125
    train = Dataset(train, 1, 0, wsize, 1, 1)
    test = Dataset(test, 1, 0, wsize, 1, 1)

    detector = Loda(wsize, 1600)

    # use train set just to get statistics in the model
    for i in range(len(train)):
        detector.add_data_point(train[i])

    # pass on test set
    scores = np.zeros(len(test.data))
    for i in range(len(test)):
        window_score = detector.add_data_point(test[i])
        # update score of elements in window
        for u in range(i, i + wsize):
            scores[u] = max(scores[u], window_score)

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def loda_machine():
    train = "../data/machine_temperature/machine_temperature_system_failure_train.csv"
    test = "../data/machine_temperature/machine_temperature_system_failure_test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    wsize = 301
    train = Dataset(train, 0, 0, wsize, 1, 1)
    test = Dataset(test, 0, 0, wsize, 1, 1)

    detector = Loda(wsize, 800)

    # use train set just to get statistics in the model
    for i in range(len(train)):
        detector.add_data_point(train[i])

    # pass on test set
    scores = np.zeros(len(test.data))
    for i in range(len(test)):
        window_score = detector.add_data_point(test[i])
        # update score of elements in window
        for u in range(i, i + wsize):
            scores[u] = max(scores[u], window_score)

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def loda_riccione():
    train = "../data/riccione/train.csv"
    test = "../data/riccione/test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    wsize = 301
    train = Dataset(train, 0, 0, wsize, 1, 1)
    test = Dataset(test, 0, 0, wsize, 1, 1)

    detector = Loda(wsize, 800)

    # use train set just to get statistics in the model
    for i in range(len(train)):
        detector.add_data_point(train[i])

    # pass on test set
    scores = np.zeros(len(test.data))
    for i in range(len(test)):
        window_score = detector.add_data_point(test[i])
        # update score of elements in window
        for u in range(i, i + wsize):
            scores[u] = max(scores[u], window_score)

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def luminol_artificial(other=False):
    train = "../data/artificial/artificial_train.csv"
    test = "../data/artificial/artificial_test1.csv"
    if other:
        test = "../data/artificial/artificial_test2.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    lag_size = 1500
    train = Dataset(train, 1, 0, lag_size, 1, 1).data
    test = Dataset(test, 1, 0, lag_size, 1, 1).data

    lumi_params = dict()
    lumi_params["precision"] = 8
    lumi_params["lag_window_size"] = lag_size
    lumi_params["future_window_size"] = 1500
    lumi_params["chunk_size"] = 7

    # put data as a dict, required by luminol
    processed_data = np.concatenate([train, test])
    ts = dict()
    for i, d in enumerate(processed_data):
        ts[i] = d
    detector = luminol.anomaly_detector.AnomalyDetector(ts, algorithm_params=lumi_params)

    # get scores
    score = detector.get_all_scores()
    scores = []
    for (timestamp, value) in score.iteritems():
        scores.append(value)
    # keep scores only related to test set
    scores = scores[len(train):]

    # normalize
    scores = np.array(scores)
    if scores.max() != 0:
        scores /= scores.max()

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def luminol_taxi():
    train = "../data/taxi/nyc_taxi_train.csv"
    test = "../data/taxi/nyc_taxi_test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    lag_size = 2000
    train = Dataset(train, 1, 1, lag_size, 1, 1).data
    test = Dataset(test, 1, 1, lag_size, 1, 1).data

    lumi_params = dict()
    lumi_params["precision"] = 4
    lumi_params["lag_window_size"] = lag_size
    lumi_params["future_window_size"] = 40
    lumi_params["chunk_size"] = 4

    # put data as a dict, required by luminol
    processed_data = np.concatenate([train, test])
    ts = dict()
    for i, d in enumerate(processed_data):
        ts[i] = d
    detector = luminol.anomaly_detector.AnomalyDetector(ts, algorithm_params=lumi_params)

    # get scores
    score = detector.get_all_scores()
    scores = []
    for (timestamp, value) in score.iteritems():
        scores.append(value)
    # keep scores only related to test set
    scores = scores[len(train):]

    # normalize
    scores = np.array(scores)
    if scores.max() != 0:
        scores /= scores.max()

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def luminol_machine():
    train = "../data/machine_temperature/machine_temperature_system_failure_train.csv"
    test = "../data/machine_temperature/machine_temperature_system_failure_test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    lag_size = 1500
    train = Dataset(train, 0, 1, lag_size, 1, 1).data
    test = Dataset(test, 0, 1, lag_size, 1, 1).data

    lumi_params = dict()
    lumi_params["precision"] = 2
    lumi_params["lag_window_size"] = lag_size
    lumi_params["future_window_size"] = 1200
    lumi_params["chunk_size"] = 5

    # put data as a dict, required by luminol
    processed_data = np.concatenate([train, test])
    ts = dict()
    for i, d in enumerate(processed_data):
        ts[i] = d
    detector = luminol.anomaly_detector.AnomalyDetector(ts, algorithm_params=lumi_params)

    # get scores
    score = detector.get_all_scores()
    scores = []
    for (timestamp, value) in score.iteritems():
        scores.append(value)
    # keep scores only related to test set
    scores = scores[len(train):]

    # normalize
    scores = np.array(scores)
    if scores.max() != 0:
        scores /= scores.max()

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


def luminol_riccione():
    train = "../data/riccione/train.csv"
    test = "../data/riccione/test.csv"
    train, _ = loader_generic(train)
    test, labels = loader_generic(test)
    lag_size = 5000
    train = Dataset(train, 1, 1, lag_size, 1, 1).data
    test = Dataset(test, 1, 1, lag_size, 1, 1).data

    lumi_params = dict()
    lumi_params["precision"] = 14
    lumi_params["lag_window_size"] = lag_size
    lumi_params["future_window_size"] = 240
    lumi_params["chunk_size"] = 15

    # put data as a dict, required by luminol
    processed_data = np.concatenate([train, test])
    ts = dict()
    for i, d in enumerate(processed_data):
        ts[i] = d
    detector = luminol.anomaly_detector.AnomalyDetector(ts, algorithm_params=lumi_params)

    # get scores
    score = detector.get_all_scores()
    scores = []
    for (timestamp, value) in score.iteritems():
        scores.append(value)
    # keep scores only related to test set
    scores = scores[len(train):]

    # normalize
    scores = np.array(scores)
    if scores.max() != 0:
        scores /= scores.max()

    beta = 1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))
    beta = 0.1
    thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc = get_thres(labels[:len(scores)], scores, beta)
    print("beta 0.1")
    print("thres %s |f1 %s |fpr %s |rfpr %s |p %s |r %s |tot_pred %s |tot_labels %s |tot_cor %s |rpc %s |roc %s" % (
        thres, f1, fpr, rfpr, p, r, tot_pred, tot_labels, tot_cor, rpc, roc))


if __name__ == "__main__":
    np.random.seed(1337)
    algo = sys.argv[1]
    dataset = sys.argv[2]

    if algo == "som":
        if dataset == "artificial1":
            som_articificial()
        elif dataset == "artificial2":
            som_articificial(True)
        elif dataset == "taxi":
            som_taxi()
        elif dataset == "machine":
            som_machine()
        elif dataset == "riccione":
            som_riccione()
    elif algo == "loda":
        if dataset == "artificial1":
            loda_artificial()
        elif dataset == "artificial2":
            loda_artificial(True)
        elif dataset == "taxi":
            loda_taxi()
        elif dataset == "machine":
            loda_machine()
        elif dataset == "riccione":
            loda_riccione()
    elif algo == "luminol":
        if dataset == "artificial1":
            luminol_artificial()
        elif dataset == "artificial2":
            luminol_artificial(True)
        elif dataset == "taxi":
            luminol_taxi()
        elif dataset == "machine":
            luminol_machine()
        elif dataset == "riccione":
            luminol_riccione()

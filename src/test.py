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


if __name__ == "__main__":
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

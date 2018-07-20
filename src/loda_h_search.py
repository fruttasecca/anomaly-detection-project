#!/usr/bin/python3
import numpy as np
import itertools

import scorer
from Loda import Loda
from dataset import Dataset
from loaders import loader_generic
from scorer import scorer_F1


def get_thres(dataset, anomalies, predictions, window_size):
    thres = np.arange(0, 1, 0.01)
    best_f1 = 0
    best_thres = 0
    # for each thres compute f1
    for th in thres:
        predicted_anomalies = []
        for idx, score in enumerate(predictions):
            if score > th:
                predicted_anomalies.extend(list(range(idx, idx + window_size)))

        precision, recall, f1 = scorer_F1(dataset, anomalies, predicted_anomalies)
        if f1 > best_f1:
            best_f1 = f1
            best_thres = th

    return best_thres, best_f1


# parameters of the search (seed, data, number of trials, etc.)
np.random.seed(1337)
data = loader_generic("../data/nyc_taxi.csv")
anomalies = scorer.nyc_taxi_anomalies
iterations = 1000

# parameters of the model
smoothing = [True, False]
normalization = [True, False]
window_size = [11, 25, 51, 75, 101, 125, 151, 175, 201, 225, 250, 275, 300]
memory = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

params = [smoothing, normalization, window_size, memory]
params = list(itertools.product(*params))
params = np.array(params)
indices = np.random.randint(0, params.shape[0], iterations)
params = params[indices]

for u in range(iterations):
    tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory = params[u]

    # init
    predictions = []
    dataset = Dataset(data, tmp_smoothing, tmp_normalization, tmp_window_size, 1, 1)
    detector = Loda(tmp_window_size, tmp_memory)

    # give score to each window
    for i in range(len(dataset)):
        data_point = dataset[i]
        score = detector.add_data_point(data_point)
        predictions.append(score)

    # check which threshold would give the best f1
    thres, f1 = get_thres(dataset, anomalies, predictions, tmp_window_size)

    print("--------")
    print("smoothing= %s, normalization= %s, wsize = %s, memory = %s thres= %s -> f1 = %s" % (
    tmp_smoothing, tmp_normalization, tmp_window_size, tmp_memory, thres, f1))

#!/usr/bin/python3
import numpy as np
import itertools

import scorer
from SomAnomalyDetector import SomAnomalyDetector
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
dimension = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
window_size = [11, 25, 51, 75, 101, 125, 151, 175, 201, 225, 250, 275, 300]
sigma = [0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5]
update_weight = [0.001, 0.005, 0.1, 0.2, 0.3, 0.4]
beta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
decay_period = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 5000, 7000]

params = [smoothing, normalization, dimension, window_size, sigma, update_weight, beta, decay_period]
params = list(itertools.product(*params))
params = np.array(params)
indices = np.random.randint(0, params.shape[0], iterations)
params = params[indices]

for u in range(iterations):
    tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight, tmp_beta, tmp_decay_period = \
    params[u]

    # init
    predictions = []
    dataset = Dataset(data, tmp_smoothing, int(tmp_normalization), int(tmp_window_size), 1, 1)
    detector = SomAnomalyDetector(int(tmp_dimension), int(tmp_dimension), int(tmp_window_size), tmp_sigma, tmp_update_weight, tmp_beta,
                                  int(tmp_decay_period), decay_factor=0.5)

    # give score to each window
    for i in range(len(dataset)):
        data_point = dataset[i]
        score = detector.add_data_point(data_point)
        predictions.append(score)

    # check which threshold would give the best f1
    thres, f1 = get_thres(dataset, anomalies, predictions, int(tmp_window_size))

    print("--------")
    msg = "smoothing= %s, normalization= %s, dimension= %s, wsize= %s, sigma= %s, update_weight= %s, beta= %s, " \
          "decay_period= %s, thres= %s -> f1 = %s"
    print(msg % (tmp_smoothing, tmp_normalization, tmp_dimension, tmp_window_size, tmp_sigma, tmp_update_weight, tmp_beta, tmp_decay_period, thres, f1))

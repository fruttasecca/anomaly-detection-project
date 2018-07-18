#!/usr/bin/python3

from loaders import loader_generic, loader_riccione
from dataset import Dataset
from sensors import Sensors
from SomAnomalyDetector import SomAnomalyDetector
import numpy as np

np.random.seed(1337)

data = loader_generic("../data/nyc_taxi.csv")
dataset = Dataset(data, False, False, 5, 1)

print(len(dataset))
print(data[:10])
print(dataset[0])
print(dataset[1])
print(dataset[3])

# test smoothing & normalize
dataset = Dataset(data, True, True, 5, 1)

print(len(dataset))
print(data[:10])
print(dataset[0])
print(dataset[1])
print(dataset[3])

# test loeader dataset riccione
data = loader_riccione('../data/2016-04-11.csv', Sensors.AMMO1)
dataset = Dataset(data, True, True, 5, 1)

print(len(dataset))
print(data[:10])
print(dataset[0])
print(dataset[1])
print(dataset[3])

######################### TAXIS
# window_size = 150
# data = loader_generic("../data/nyc_taxi.csv")
# dataset = Dataset(data, False, False, window_size, 1)
# detector = SomAnomalyDetector(10, 10, window_size, sigma=1.0, update_weight=0.1, distance_thres=0.30, frequency_thres=100, ignored_items=300, decay_period=5000, decay_factor=0.5)
# # detector = SomAnomalyDetector(10, 10, window_size, sigma=1.0, update_weight=0.1, distance_thres=0.30, frequency_thres=100, ignored_items=300, decay_period=5000, decay_factor=0.5)
# #             no    no   no     si    si    si    si
# # anomalies: [523, 908, 3935, 8465, 8787, 9929, 10080]
#
# anomalies = []
# for i in range(len(dataset)):
#     data_point = dataset[i]
#     anomaly, dist = detector.add_data_point(data_point)
#     if anomaly and (len(anomalies) == 0 or anomalies[-1] < i - window_size):
#         anomalies.append(i)
#         print(i)
#         print(dist)
# print(len(anomalies))
# print(anomalies)


######################### MACHINE TEMPERATURE
# window_size = 100
# data = loader_generic("../data/machine_temperature_system_failure.csv")
# dataset = Dataset(data, False, False, window_size, 1)
#
# detector = SomAnomalyDetector(10, 10, window_size, sigma=1.0, update_weight=0.1, distance_thres=0.10, frequency_thres=100, ignored_items=300, decay_period=5000, decay_factor=0.5)

# con smoothing,
# window_size = 101
# dataset = Dataset(data, True, False, window_size, 1, 1)
# detector = SomAnomalyDetector(10, 10, window_size, sigma=1.0, update_weight=0.1, distance_thres=0.05, frequency_thres=100, ignored_items=300, decay_period=5000, decay_factor=0.5)
#
# anomalies = []
# for i in range(len(dataset)):
#     data_point = dataset[i]
#     anomaly, dist = detector.add_data_point(data_point)
#     if anomaly and (len(anomalies) == 0 or anomalies[-1] < i - window_size):
#         anomalies.append(i)
#         print(i)
#         print(dist)
# print(len(anomalies))
# print(anomalies)

# TAXI, WITH SMOOTHING, window_size 51, window warmup 700, 0.30 thres
# detector = Loda(window_size, 700, 0.30)
# [3211, 7160, 8457, 8509, 8649, 8985, 10030, 10082]

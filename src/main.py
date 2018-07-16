#!/usr/bin/python3

from loaders import loader_generic, loader_riccione
from dataset import Dataset
from sensors import Sensors


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


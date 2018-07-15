#!/usr/bin/python3

from data_loader_generic import loader_generic
from dataset import Dataset


data = loader_generic("../data/nyc_taxi.csv")
dataset = Dataset(data, False, False, 5, 1)

print(len(dataset))
print(data[:10])
print(dataset[0])
print(dataset[1])
print(dataset[3])



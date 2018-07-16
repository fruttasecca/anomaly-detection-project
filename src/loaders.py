"""
Data loader class for generic (NAB) datasets, which are in date - real value format.
"""

import pandas as pd


def loader_generic(file):
    """
    Given a file with 2 columns, 1 for data 1 for value, return a numpy ndarry of values from the second column, keeping
    order.
    :param file: Csv file with 2 columns.
    :return: Numpy ndarray of values in the second column of the file.
    """
    data = pd.read_csv(file)
    return data["value"].values


def loader_riccione(file, sensor):
    """
    loads file of data of one sensor from riccione dataset in a vector
    :param file: input file
    :param sensor: sensor is a variable of sensor type (see sensors.py)
    """
    data = pd.read_csv(file, header=None, delimiter=';')
    return data[data[1] == sensor.string][2].fillna(0).values

import math
import numpy as np
from numpy import (power, exp, arange, outer)


class Som(object):
    def __init__(self, x, y, vector_size, sigma=0.3, update_weight=0.20):
        """
        Initializes the self organizing map. You can pass functions to customize how
        representants are initialized, how the distance between a data_point and a representant
        is computed, and how a representant should be updated.

        :param x: Width of the som.
        :param y: Height of the som.
        :param vector_size: Length of each data point, which is a vector; each representant in the SOM will be
        a vector of the same length.
        :param sigma: Sigma parameter for the gaussian neighbourhood function.
        :param update_weight: Weight for adding a part of the difference between the data point and its rep to the
        rep.
        """
        self._x = x
        self._y = y
        self._vector_size = vector_size
        self._sigma = sigma
        self._update_weight = update_weight

        self._constructor = lambda *args: np.random.random((1, self._vector_size))
        self._distance = lambda data_point, representant: np.linalg.norm(
            data_point / np.linalg.norm(data_point) - representant / np.linalg.norm(representant))
        self._update_function = lambda input, rep, weight: rep + ((input - rep) * weight * self._update_weight)

        # init representants
        self._reps = np.empty((x, y), dtype=object)
        for i in range(x):
            for j in range(y):
                self._reps[i][j] = self._constructor()

        # to avoid calling arange every time we compute the gaussian (see self._gaussian_neigh...)
        self._neighx = arange(x)
        self._neighy = arange(y)
        self._d = self._sigma * self._sigma

    def _gaussian_neighbourhood(self, rep_cords):
        """
        Compute the gaussian similarities between the coordinates of the representant
        and the coordinates of the rest of the map.
        :param rep_cords: Coordinates of the representant.
        :return: A matrix containing the gaussian similarities with the coordinates
        of the representant.
        """
        ax = exp(-power(self._neighx - rep_cords[0], 2) / self._d)
        ay = exp(-power(self._neighy - rep_cords[1], 2) / self._d)
        return outer(ax, ay)

    def _update(self, data_point, representant):
        """
        Update the representants based on their similarity with the input. The most
        similar representant (similar to the input) will be updated the most, the update
        will "flood" starting from that representant, slowly losing weight.
        :param data_point: Data_point based on which we will update the reps.
        :param representant: Representant of the data_point (pair of coordinates), from where
        the update flood will start from.
        """
        neigh_dist = self._gaussian_neighbourhood(representant)
        for i in range(self._x):
            for j in range(self._y):
                self._reps[i][j] = self._update_function(data_point, self._reps[i][j], neigh_dist[i][j])

    def get_representant(self, data_point):
        """
        Get the representant (as coordinates) and the distance between the representant and the
        data_point.
        :param data_point: Data_point for which we want the rep and the distance.
        :return: A (representant, distance) pair, where the first element
        are the 2 coordinates (x,y) of the representant with the lowest distance, the
        second element is the distance itself.
        """
        distance = float('inf')
        rep = None
        for i in range(self._x):
            for j in range(self._y):
                tmp = self._distance(self._reps[i][j], data_point)
                if distance > tmp:
                    distance = tmp
                    rep = (i, j)

        return rep, distance

    def add_data_point(self, data_point):
        """
        Add a data point to the map, updating the representants based on distances between
        them and the data_point.
        :param data_point: Data point to add.
        :return: A (representant, distance) pair, where the first element
        are the 2 coordinates (x,y) of the representant with the lowest distance, the
        second element is the distance itself.
        """
        rep, distance = self.get_representant(data_point)
        self._update(data_point, rep)
        return rep, distance


class SomAnomalyDetector(object):
    """
    Base class for Som based clusterers.
    """

    def __init__(self, x, y, vector_size, sigma, update_weight, beta, ignored_items=0,
                 decay_period=20000, decay_factor=0.7):
        """
        Initializes the clusterer by initializing the Som. This can be used without providing the distance or update function, but
        if those are not provided default functions that assume inputs to be np.arrays are going to be used. A constructor
        function for representants is to be provided.

        :param x: Width of the som.
        :param y: Height of the som.
        :param vector_size: Length of each data point, which is a vector; each representant in the SOM will be
        a vector of the same length.
        :param sigma: Sigma parameter for the gaussian neighbourhood function for the Som.
        :param ignored_items: Number of items to ignore, the ignored items will still count towards updating the som, but
        will have an anomaly score of 0.
        :param decay_period: Every decay_period frequencies will decay based on the decay factor.
        :param decay_factor: Frequencies will me multiplied by this value every decary_period.
        """
        # self organizing map
        self._som = Som(x, y, vector_size, sigma, update_weight)

        self._beta = beta

        self._ignored_items = ignored_items
        self._decay_period = decay_period
        self._decay_factor = decay_factor

        '''
        Init the table of counters to keep track of how many times we have seen something relatable to a certain
        representant recently, the table will have the same shape of the Som.
        Each frequency is periodically multiplied by a decaying factor.
        '''
        self._frequencies = np.zeros((x, y))
        self._counter = 0

    def get_representants(self):
        return self._som._reps

    def get_frequencies(self):
        return self._frequencies

    def add_data_point(self, data_point):
        """
        Clusters an item, returning its representant and if it is an anomaly or not.
        :param data_point: Item to add.
        :return: Anomaly score of the item.
        """
        rep, distance_score = self._som.add_data_point(data_point)
        self._ignored_items = max(-1, self._ignored_items - 1)
        self._counter += 1

        # update and get
        freq = self._frequencies[rep] = self._frequencies[rep] + 1

        # decay frequencies if we reached the end of our decay period
        if self._counter == self._decay_period:
            self._counter *= self._decay_factor
            self._frequencies *= self._decay_factor

        # negative log of frequency
        frequency_score = -math.log(freq / self._counter)

        # combine scores
        total_score = (1. - self._beta) * distance_score + self._beta * frequency_score
        if self._ignored_items > 0:
            total_score = 0
        return total_score

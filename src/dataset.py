"""
Dataset class
"""
import numpy as np


class Dataset(object):
    def __init__(self, data, do_smoothing, do_norm, window_size, step_size):
        """
        Dataset class, contains data and return windows from it.
        :param data: A numpy array.
        :param do_smoothing: If smoothing has to be done before returning a window, only considering values in that window.
        :param do_norm: If normalization has to be done before returning a window, only considering values in that window.
        :param window_size: Size of the window, given an index, values [I, I + windoww_size -1] will be retrieved.
        :param step_size: Size of the step of data points between each window.
        """

        self.data = data
        self.do_smooothing = do_smoothing
        self.do_norm = do_norm
        self.window_size = window_size
        self.step_size = step_size

    def __len__(self):
        return len(self.data) // self.step_size - self.window_size // self.step_size + 1

    @staticmethod
    def __smoothing__(data):
        print("MEMEXCEPTION, PLS IMPLEMENT ME")
        exit()
        return data

    @staticmethod
    def __normalize__(data):
        """
        Normalize vector.
        :param data: Numpy vector to normalize.
        :return: Normalized vector.
        """
        return data / np.linalg.norm(data)

    def __getitem__(self, window_index):
        """
        Given a window index, retrieve window.
        :param window_index: Index of a window.
        :return: Numpy array containing a window of data points.
        """
        if window_index >= self.__len__():
            print("Exceeding length")
            exit()
        start_idx = self.step_size * window_index
        end_idx = start_idx + self.window_size
        window = self.data[start_idx: end_idx]

        if self.do_smooothing:
            window = self.__smoothing__(window)

        if self.do_norm:
            window = self.__normalize__(window)

        return window

    def indexes_of(self, window_index):
        """
        Return what would be the left and right index of a certain window
        in the data, i.e. the first window, having window_index = 0, would have a left index of 0
        and a right index of self.window_size.
        :param window_index: Index of a window.
        :return: Tuple containing left and right index of data for this window, [left, right).
        """
        if window_index >= self.__len__():
            print("Exceeding length")
            exit()
        start_idx = self.step_size * window_index
        end_idx = start_idx + self.window_size
        return start_idx, end_idx



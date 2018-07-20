import numpy as np
import math


class Loda(object):
    def __init__(self, vector_size, window_size):
        self._dimensions = vector_size
        self.window_size = window_size

        self._current_point = 0
        self._window_data = []
        self._vectors = None
        self._histograms = None
        self._bin_sizes = None
        self._min_vals = None
        self._max_vals = None

        # hard limits
        self._max_histograms = 1000
        self._max_bins = int((self.window_size / math.log(self.window_size)))
        self._tau = 0.01

        # non zero elements of vectors for projections
        self._nonzero_mask = np.zeros((1, self._dimensions))
        self._nonzero_mask[0, :int(math.sqrt(self._dimensions))] = 1

    def add_data_point(self, data_point):
        """
        Add a data point, returning if it is an anomaly and its anomaly score.
        :param data_point:
        :return:
        """
        if self._current_point == self.window_size:
            self._vectors, self._histograms, self._bin_sizes, self._min_vals, self._max_vals = self._build_histograms()
            # reset counter and saved data points
            self._current_point = 0
            self._window_data = []

        self._current_point += 1
        self._window_data.append(data_point)

        anomaly_score = 0
        # if we are not saving data points to build vectors and histograms for the first time
        if self._histograms is not None:
            anomaly_score = self._score(data_point)

        return anomaly_score

    def _score(self, data_point):
        """
        Score the sample using histograms.
        :param data_point:
        :return:
        """
        # make projections and normalize
        dot_prods = (np.dot(self._vectors, data_point) - self._min_vals) / (self._max_vals - self._min_vals)

        # get probabilities from histograms
        for i, histo in zip(range(dot_prods.size), self._histograms):
            dot_prods[i] = histo.get(dot_prods[i] // self._bin_sizes[i], 1e-4)

        # compute score, normalize by dimensions
        score = (np.sum(-np.log(dot_prods / self.window_size))) / (len(dot_prods) * self._dimensions)
        return score

    def _score_contribution(self, vector, histogram, bin_size, min_val, max_val, data_points):
        """
        Get score contribution for all samples from a single histogram.
        :param vector:
        :param histogram:
        :param bin_size:
        :param min_val:
        :param max_val:
        :param data_points:
        :return:
        """
        # make projections and normalize
        dot_prods = (np.dot(vector, data_points) - min_val) / (max_val - min_val)

        # get probabilities from histograms
        for i in range(self.window_size):
            dot_prods[0, i] = histogram.get(dot_prods[0, i] // bin_size, 1e-4)

        # compute score, normalize by number of dimensions
        dot_prods = (-np.log(dot_prods / self.window_size)) / self._dimensions
        return dot_prods

    def _build_histograms(self):
        """
        Iteratively build histograms until the reduction of variance is below a certain threshold, section
        4.1.1 of LODA paper.
        :return:
        """
        vectors = []
        histograms = []
        bins_sizes = []
        min_vals = []
        max_vals = []
        data = np.column_stack(self._window_data)

        # get scores from vector 1
        self._add_new_histogram(vectors, histograms, bins_sizes, min_vals, max_vals, data)
        scores1 = self._score_contribution(vectors[-1], histograms[-1], bins_sizes[-1], min_vals[-1], max_vals[-1],
                                           data)

        # get scores from vector 2
        self._add_new_histogram(vectors, histograms, bins_sizes, min_vals, max_vals, data)
        scores_cumulative = self._score_contribution(vectors[-1], histograms[-1], bins_sizes[-1], min_vals[-1],
                                                     max_vals[-1], data)
        scores_cumulative += scores1
        # compute the first alpha, used to normalize future alphas
        alpha_one = np.sum(np.fabs(scores_cumulative / 2 - scores1)) / self.window_size

        for i in range(2, self._max_histograms):

            self._add_new_histogram(vectors, histograms, bins_sizes, min_vals, max_vals, data)
            # score contribution from the last generated histogram
            scores_tmp = self._score_contribution(vectors[-1], histograms[-1], bins_sizes[-1], min_vals[-1],
                                                  max_vals[-1], data)

            # scores of f_k
            scores_old = scores_cumulative
            # scores of f_k+1
            scores_cumulative = scores_old + scores_tmp

            alpha = np.sum(np.fabs((scores_cumulative / (i + 1) - scores_old / i))) / self.window_size
            if alpha / alpha_one <= self._tau:
                break

        # return histograms
        vectors = np.row_stack(vectors[:i])
        bin_sizes = np.asarray(bins_sizes[:i])
        min_vals = np.asarray(min_vals[:i])
        max_vals = np.asarray(max_vals[:i])
        return vectors, histograms[:i], bin_sizes, min_vals, max_vals

    def _add_new_histogram(self, vectors, histograms, bins_sizes, min_vals, max_vals, data):
        """
        Add a new histogram, selecting the number of bins that maximizes the penalized
        maximum likelihood (Birgé and Rozenhole (2006)).
        :param vectors:
        :param histograms:
        :param bins_sizes:
        :param min_vals:
        :param max_vals:
        :param data:
        :return:
        """
        # random vector with sqrt(dimensions) non-zero elements
        vector = np.random.random((1, self._dimensions))
        np.random.shuffle(self._nonzero_mask[0])
        vector *= self._nonzero_mask

        # compute scores once
        scores = np.dot(vector, data)
        max_val = scores.max()
        min_val = scores.min()
        scores = (scores - min_val) / (max_val - min_val)

        # check which number of bins is the best (maximizing penalized maximum likelihood, birgè and rozenhole 2006)
        best_bin_number_score = -1e40
        best_bin_number = 0
        best_histogram = None

        for b in range(1, self._max_bins):
            histogram = dict()
            bin_width = 1 / b

            # put scores in histogram
            for i in range(scores.size):
                score = scores[0, i]
                index = score // bin_width
                histogram[index] = histogram.get(index, 0) + 1

            # get the score for this number of bins
            b_score = -b + 1 - math.log(b) ** 2.5
            for value in histogram.values():
                # += examples in that bin + log(b * examples in that bin / total examples in histogram)
                b_score += value * math.log(b * value / self.window_size)

            if b_score > best_bin_number_score:
                best_bin_number_score = b_score
                best_bin_number = b
                best_histogram = histogram
        # append results to list
        vectors.append(vector)
        histograms.append(best_histogram)
        bins_sizes.append(1 / best_bin_number)
        min_vals.append(min_val)
        max_vals.append(max_val)

"""
File containing what indexes in the different datasets are to be considered anomalies, and a class
to score models prediction (F1).
"""

nyc_taxi_anomalies = list(range(5838, 6046)) + list(range(7080, 7287)) + list(range(8423, 8629)) + list(
    range(8731, 8938)) + list(range(9977, 10184))

nyc_taxi_size = 10320
machine_temperature_anomalies = list(range(2126, 2691)) + list(range(3702, 4268)) + list(range(16056, 16622)) + list(range(19231, 19797))
machine_temperature_size = 22685


def scorer_F1(dataset, anomalies, predicted_anomalies):
    """
    Given the size of the data, the indexes that are to be considered as anomalies,
    and indexes that represent the predicted anomalies, compute the F1 score.
    The intended use is to use the anomalies lists in this module as "anomalies",
    choosing a different one depending on the dataset, as in:
    scorer_F1(nyc_taxi_size, nyc_taxi_anomalies, predicted).

    :param dataset:
    :param anomalies: List of indexes that are anomalies, indexes that are not part of the indexes
    of the dataset are going to be ignored, this allows you to use a dev or training dataset sampled from the
    whole dataset, and just pass along all the anomalies, this will make sure only the anomalies in the passed
    dataset are considered; i.e. scorer_F1(dev_set, nyc_taxi_anomalies (all anomalies), predictions).
    :param predicted_anomalies: List of indexes that are predicted to be anomalies.
    :return: F1 score.
    """
    size = len(dataset.data)
    anomalies = set(anomalies).intersection(set(list(range(size))))
    predicted_anomalies = set(predicted_anomalies)
    not_anomalies = set(list(range(size))).difference(anomalies)
    not_predicted_anomalies = set(list(range(size))).difference(predicted_anomalies)
    true_positive = len(anomalies & predicted_anomalies)
    false_positive = len(predicted_anomalies & not_anomalies)
    false_negative = len(not_predicted_anomalies & anomalies)
    # true_negative = len(not_predicted_anomalies & not_anomalies)

    precision = true_positive / (true_positive + false_positive + 1e-7)
    recall = true_positive / (true_positive + false_negative + 1e-7)
    f1 = (2 * (precision * recall)) / (precision + recall + 1e-7)
    return precision, recall, f1



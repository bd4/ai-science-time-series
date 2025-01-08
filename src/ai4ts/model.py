import abc


class Forecast(object):
    def __init__(self, data, name, model=None):
        self.data = data
        self.name = name
        self.model = model

    def __len__(self):
        return len(self.data)


class TimeSeriesModel(abc.ABC):
    """
    Flag for deep learning and other flexible models, indicating not to apply custom
    per-dataset pre/post processing (e.g. taking log of the data).
    """
    deep = True

    @abc.abstractmethod
    def fit(self, times, history, max_prediction_length, **kwargs):
        """
        Fit a univariate times series model model with the specified times
        and values and max prediction horizon.

        :param times: numpy array-like of times
        :param history: numpy array-like of time series values

        :returns: None

        """
        ...

    @abc.abstractmethod
    def predict(self, prediction_length):
        """
        Use fitted model to predict future values.

        :param n: number of points to predict

        :returns: ai4ts.model.Forecast object
        """
        ...

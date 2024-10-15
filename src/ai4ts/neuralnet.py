#!/usr/bin/env python3

import torch
import numpy as np
import pandas as pd
# import torch.nn.functional as F

import ai4ts


def get_lag_input_output_sizes(series_len, lag):
    in_size = (series_len - lag, lag)
    out_size = (series_len - lag,)
    return (in_size, out_size)


def create_lag_input_output(series_data, lag):
    """
    Set up arrays for training an autoregressive neural network using
    the specified lag. Lag 1 correspends to an AR(1), etc.
    """
    series_len = len(series_data)
    in_arrays = [torch.tensor(series_data[i - lag : i]) for i in range(lag, series_len)]
    in_matrix = torch.stack(in_arrays, 0)
    out_vector = torch.tensor([series_data[i] for i in range(lag, series_len)])
    return (in_matrix, out_vector)


def mean_sq_error(predictions, actual):
    return torch.square(predictions - actual).mean()


class SimpleLinearModel(ai4ts.model.TimeSeriesModel):
    def fit(self, times, history, max_prediction_length, **kwargs):
        if isinstance(history, pd.Series):
            history = history.values
        self.order = kwargs.get("order")
        if self.order is None:
            ar_order = kwargs.get("ar_order", 0)
            i_order = kwargs.get("i_order", 0)
            ma_order = kwargs.get("ma_order", 0)
            self.order = (ar_order, i_order, ma_order)
        self.lag = self.order[0]
        self.init(len(history), self.lag, n_hidden=10, lr=0.00002)
        in_matrix, out_vector = create_lag_input_output(history, self.lag)
        self.train(in_matrix, out_vector, epochs=100)
        self.indep = history[-self.lag :]

    def predict(self, prediction_length):
        prediction = np.ndarray(prediction_length)
        indep = torch.tensor(self.indep)
        with torch.no_grad():
            for i in range(prediction_length):
                prediction[i] = self._predict(indep)
                indep = indep.roll(1)
                indep[-1] = prediction[i]
        return ai4ts.model.Forecast(prediction, "Simple Linear", self)

    def init(self, series_len, lag, n_hidden, lr):
        self.lr = lr
        self.layer1 = torch.ones(lag, 1) / lag
        self.layer1.requires_grad_()
        self.bias = torch.ones(1)[0]
        self.bias.requires_grad_()

    def _predict(self, indep):
        return indep @ self.layer1 + self.bias
        # res = F.relu(indep @ self.layer1)
        # res = res @ self.layer2 + self.bias
        # return torch.sigmoid(res)

    def get_loss(self, independent, dependent):
        return mean_sq_error(self._predict(independent), dependent)

    def _update_coeffs(self):
        for layer in (self.layer1, self.bias):
            layer.sub_(layer.grad * self.lr)
            layer.grad.zero_()

    def _step(self, independent, dependent):
        loss = self.get_loss(independent, dependent)
        loss.backward()
        with torch.no_grad():
            self._update_coeffs()
        print(f"{loss:.3f}", end="; ")

    def train(self, xs, y, epochs=30):
        # import ipdb; ipdb.set_trace()
        for i in range(epochs):
            self._step(xs, y)


if __name__ == "__main__":
    series = np.array(range(100), dtype=np.float32)
    lag = 1
    m = SimpleLinearModel()
    m.fit(None, series, 2, order=(lag, 0, 0))
    fcast = m.predict(2)
    print(fcast.data)

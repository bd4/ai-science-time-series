#!/usr/bin/env python3

import math

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

import ai4ts

import ipdb


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
    l = (predictions - actual) ** 2 / 2
    return l.mean()


class SimpleLinearRegression(object):
    def __init__(self, lag, batch_size, lr, sigma=0.01, grad_clip=None):
        self.lag = lag
        self.lr = lr
        self.sigma = sigma
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.layer1 = torch.normal(0, sigma, (lag, 1))
        # self.layer1 = torch.ones((lag, 1)) / lag
        self.layer1.requires_grad_()
        self.bias = torch.zeros(1)
        self.bias.requires_grad_()
        self.best_loss = None
        self.best_params = None

    def predict(self, indep):
        return torch.matmul(indep, self.layer1) + self.bias

    def get_loss(self, independent, dependent):
        return mean_sq_error(self.predict(independent).t(), dependent)

    def _update_coeffs(self):
        for layer in (self.layer1, self.bias):
            grad = layer.grad
            if self.grad_clip is not None:
                # TODO: clip on all params together?
                grad_norm = torch.sqrt(torch.sum(grad**2))
                if grad_norm > self.grad_clip:
                    grad = grad * self.grad_clip / grad_norm
            layer.sub_(grad * self.lr)
            layer.grad.zero_()

    def _get_batch(self, inputs, outputs):
        assert inputs.shape[0] == outputs.shape[0]
        idx = torch.randperm(inputs.shape[0])[: self.batch_size]
        return (inputs[idx], outputs[idx])

    def _step(self, independent, dependent):
        loss = self.get_loss(independent, torch.t(dependent))
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss.clone().detach()
            self.best_params = (
                self.layer1.clone().detach(),
                self.bias.clone().detach(),
            )
        loss.backward()
        with torch.no_grad():
            self._update_coeffs()
        # print(f"{loss:.3f} ", repr(self))

    def train(self, xs, y, epochs=30):
        # import ipdb; ipdb.set_trace()
        for i in range(epochs):
            batch_in, batch_out = self._get_batch(xs, y)
            self._step(batch_in, batch_out)
        self.layer1, self.bias = self.best_params

    def __repr__(self):
        return f"SimpleLinearRegression({self.layer1} + {self.bias})"


class SimpleNNRegression(object):
    def __init__(self, lag, batch_size, lr, n_hidden=10, sigma=0.01, grad_clip=None):
        self.lag = lag
        self.lr = lr
        self.sigma = sigma
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.layer1 = torch.normal(0, sigma, (lag, n_hidden))
        # self.layer1 = torch.ones((lag, 1)) / lag
        self.layer1.requires_grad_()
        self.layer2 = torch.normal(0, sigma, (n_hidden, 1))
        self.layer2.requires_grad_()
        self.bias = torch.ones(1) / 2.0
        self.bias.requires_grad_()
        self.best_loss = None
        self.best_params = None

    def predict(self, indep):
        res = F.relu(torch.matmul(indep, self.layer1))
        res = torch.matmul(res, self.layer2) + self.bias
        return res
        # res = F.relu(indep @ self.layer1)
        # res = res @ self.layer2 + self.bias
        # return torch.sigmoid(res)

    def get_loss(self, independent, dependent):
        return mean_sq_error(self.predict(independent).t(), dependent)

    def _update_coeffs(self):
        for layer in (self.layer1, self.layer2, self.bias):
            grad = layer.grad
            if self.grad_clip is not None:
                # TODO: clip on all params together?
                grad_norm = torch.sqrt(torch.sum(grad**2))
                if grad_norm > self.grad_clip:
                    grad = grad * self.grad_clip / grad_norm
            layer.sub_(grad * self.lr)
            layer.grad.zero_()

    def _get_batch(self, inputs, outputs):
        assert inputs.shape[0] == outputs.shape[0]
        idx = torch.randperm(inputs.shape[0])[: self.batch_size]
        return (inputs[idx], outputs[idx])

    def _step(self, independent, dependent):
        loss = self.get_loss(independent, torch.t(dependent))
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss.clone().detach()
            self.best_params = (
                self.layer1.clone().detach(),
                self.layer2.clone().detach(),
                self.bias.clone().detach(),
            )
        loss.backward()
        with torch.no_grad():
            self._update_coeffs()
        # print(f"{loss:.3f}", end="; ")

    def train(self, xs, y, epochs=30):
        # import ipdb; ipdb.set_trace()
        for i in range(epochs):
            batch_in, batch_out = self._get_batch(xs, y)
            self._step(batch_in, batch_out)
        self.layer1, self.layer2, self.bias = self.best_params

    def __repr__(self):
        return f"SimpleNNRegression({self.layer1} + {self.bias})"


def get_diff_with_tail(series, ndiff):
    tail = np.zeros(ndiff)
    for i in range(0, ndiff):
        tail[ndiff - i - 1] = series[-1]
        series = np.diff(series)
    return series, tail


def get_base_and_diff(series, ndiff):
    undiff_base = np.zeros(ndiff)
    for i in range(0, ndiff):
        undiff_base[ndiff - i - 1] = series[0]
        series = np.diff(series)
    return undiff_base, series


def undiff(base, series):
    for b in base:
        series = np.concatenate(([b], np.cumsum(series) + b))
    return series


def undiff_new_base(base, value):
    new_base = np.zeros(len(base))
    for i, b in enumerate(base):
        value += b
        new_base[i] = value
    return new_base, value


class SimpleLinearModel(ai4ts.model.TimeSeriesModel):
    model_class = SimpleLinearRegression

    def fit(self, times, history, max_prediction_length, **kwargs):
        if isinstance(history, pd.Series):
            history = history.values
        self.order = kwargs.get("order")
        if self.order is None:
            ar_order = kwargs.get("ar_order", 0)
            i_order = kwargs.get("i_order", 0)
            ma_order = kwargs.get("ma_order", 0)
            self.order = (ar_order, i_order, ma_order)
        # TODO: handle differencing
        if self.order[1] > 0:
            history, self.undiff_base = get_diff_with_tail(history, self.order[1])
        else:
            self.undiff_base = None
        self.lag = max(self.order[0], self.order[2])
        self.batch_size = int(math.ceil(len(history) / 10.0))
        self.model = self.model_class(self.lag, batch_size=self.batch_size, lr=0.00003)
        in_matrix, out_vector = create_lag_input_output(history, self.lag)
        self.model.train(in_matrix, out_vector, epochs=100)
        self.indep = history[-self.lag :]

    def predict(self, prediction_length):
        prediction = np.ndarray(prediction_length)
        indep = torch.tensor(self.indep)
        with torch.no_grad():
            for i in range(prediction_length):
                prediction[i] = self.model.predict(indep)
                indep = indep.roll(1)
                indep[-1] = prediction[i]
        if self.undiff_base is not None:
            for i in range(prediction_length):
                self.undiff_base, prediction[i] = undiff_new_base(
                    self.undiff_base, prediction[i]
                )
        return ai4ts.model.Forecast(prediction, self.model_class.__name__, self.model)


class SimpleNNModel(SimpleLinearModel):
    model_class = SimpleNNRegression


if __name__ == "__main__222":
    # series = np.array(range(100), dtype=np.float32)
    series = np.random.normal(0, 0.15, 1000).astype(np.float32)
    lag = 1
    m = SimpleLinearModel(SimpleLinearRegression)
    m.fit(None, series, 2, order=(lag, 0, 0))
    fcast = m.predict(2)
    print(fcast.data)
    print(repr(fcast.model))

    m = SimpleLinearModel(SimpleNNRegression)
    m.fit(None, series, 2, order=(lag, 0, 0))
    fcast = m.predict(2)
    print(fcast.data)
    print(repr(fcast.model))

    series2 = np.ones(1000, dtype=np.float32) * 17
    lag = 1
    m2 = SimpleLinearModel(SimpleLinearRegression)
    m2.fit(None, series2, 2, order=(lag, 0, 0))
    fcast2 = m2.predict(2)
    print(fcast2.data)
    print(repr(fcast2.model))

    m2 = SimpleLinearModel(SimpleNNRegression)
    m2.fit(None, series2, 2, order=(lag, 0, 0))
    fcast2 = m2.predict(2)
    print(fcast2.data)
    print(repr(fcast2.model))

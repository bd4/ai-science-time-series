#!/usr/bin/env python3

from itertools import islice
import os.path

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

import pandas as pd  # requires: pip install pandas
import torch
from chronos import ChronosPipeline

import ai4ts


def forecast(
    times, history, prediction_length, ar_order, i_order, ma_order, device="cuda"
):
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # forecast shape: [num_series, num_samples, prediction_length]
    forecast = pipeline.predict(
        context=torch.tensor(history),
        prediction_length=prediction_length,
        num_samples=100,
    )

    forecast_mean = forecast[0].mean(0)

    return ai4ts.model.Forecast(forecast_mean, "chronos", forecast)

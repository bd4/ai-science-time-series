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


def main(inpath, outpath, prediction_length = 24):
    values = np.loadtxt(inpath, dtype=np.float32)
    train = values[:-prediction_length]
    df = pd.DataFrame(values)
    n_cycles = len(df) / prediction_length

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
        torch_dtype=torch.bfloat16,
    )

    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # forecast shape: [num_series, num_samples, prediction_length]
    forecast = pipeline.predict(
        context=torch.tensor(train),
        prediction_length=prediction_length,
        num_samples=len(train),
    )

    history_tail = values[-min(4*prediction_length, len(values)):]
    forecast_index = range(len(history_tail) - prediction_length, len(history_tail))
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(history_tail, color="royalblue", label="historical data")
    plt.plot(forecast_index, median, color="tomato", label="median forecast")
    plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig(outpath)


if __name__ == "__main__":
    import sys
    inpath = sys.argv[1]
    outpath, _ = os.path.splitext(inpath)
    outpath += "-chronos.pdf"
    main(inpath, outpath)
    print(outpath)

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

from lag_llama.gluon.estimator import LagLlamaEstimator


def get_lag_llama_predictions(
    dataset,
    prediction_length,
    device,
    context_length=32,
    use_rope_scaling=False,
    num_samples=100,
):
    ckpt = torch.load(
        "lag-llama.ckpt", map_location=device
    )  # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(
            1.0, (context_length + prediction_length) / estimator_args["context_length"]
        ),
    }

    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,  # Lag-Llama was trained with a context length of 32, but can work with any context length
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
        batch_size=1,
        num_parallel_samples=100,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


def main(inpath, outpath):
    values = np.loadtxt(inpath, dtype=np.float32)
    df = pd.DataFrame(
        {"target": values},
        index=pd.date_range("2024-01-01", periods=len(values), freq="1H"),
    )
    dataset = PandasDataset(df)
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 100  # number of samples sampled from the probability distribution for each timestep
    device = torch.device(
        "cuda:0"
    )  # You can switch this to CPU or other GPUs if you'd like, depending on your environment

    forecasts, tss = get_lag_llama_predictions(
        dataset, prediction_length, device, num_samples
    )

    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter("%b, %d")
    plt.rcParams.update({"font.size": 15})

    # Iterate through the first 9 series, and plot the predicted samples
    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx + 1)

        plt.plot(
            ts[-4 * prediction_length :].to_timestamp(),
            label="target",
        )
        forecast.plot(color="g")
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        ax.set_title(forecast.item_id)

    plt.gcf().tight_layout()
    plt.legend()
    plt.savefig(outpath)


if __name__ == "__main__":
    import sys

    inpath = sys.argv[1]
    outpath, _ = os.path.splitext(inpath)
    outpath += ".pdf"
    main(inpath, outpath)
    print(outpath)

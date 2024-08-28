from itertools import islice
import os.path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch

from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.pandas import PandasDataset

from lag_llama.gluon.estimator import LagLlamaEstimator

import ai4ts


def get_lag_llama_predictions(
    dataset,
    prediction_length,
    device,
    context_length=32,
    use_rope_scaling=False,
    num_samples=10,
):
    ckpt_path = ai4ts.get_model_path("lag-llama.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(
            1.0,
            (context_length + prediction_length) / estimator_args["context_length"],
        ),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        # Lag-Llama was trained with a context length of 32, but can work with any context length
        context_length=context_length,
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


def forecast(
    times, history, prediction_length, ar_order, i_order, ma_order, device="cuda"
):
    # Note: lag-llama requires times as index
    df = pd.DataFrame({"target": history}, index=times)
    dataset = PandasDataset(df)
    device = torch.device(device)

    forecasts, _ = get_lag_llama_predictions(
        dataset=dataset,
        prediction_length=prediction_length,
        device=device,
        context_length=32,
        num_samples=100,
    )
    return ai4ts.model.Forecast(forecasts[0].mean, name="lag-llama", model=forecasts[0])

#!/usr/bin/env python3

from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions

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


def get_ts_dataset():
    import pandas as pd
    from gluonts.dataset.pandas import PandasDataset

    url = (
        "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
        "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
    )
    df = pd.read_csv(url, index_col=0, parse_dates=True)

    # Set numerical columns as float32
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("float32")

    # Create the Pandas
    return PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")


def main():
    backtest_dataset = get_ts_dataset()
    prediction_length = 24  # Define your prediction length. We use 24 here since the data is of hourly frequency
    num_samples = 100  # number of samples sampled from the probability distribution for each timestep
    device = torch.device(
        "cuda:0"
    )  # You can switch this to CPU or other GPUs if you'd like, depending on your environment

    forecasts, tss = get_lag_llama_predictions(
        backtest_dataset, prediction_length, device, num_samples
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
    plt.savefig("demo1.pdf")


if __name__ == "__main__":
    main()

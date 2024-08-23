#!/usr/bin/env python3

from itertools import islice
import os.path

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch

from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.pandas import PandasDataset

from lag_llama.gluon.estimator import LagLlamaEstimator

import ai4ts


def get_script_relative_path(fname):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    print("script dir ", script_dir)
    return os.path.join(script_dir, fname)


def get_lag_llama_predictions(
    dataset,
    prediction_length,
    device,
    context_length=32,
    use_rope_scaling=False,
    num_samples=10,
):
    ckpt_path = get_script_relative_path("lag-llama.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(
            1.0,
            (context_length + prediction_length)
            / estimator_args["context_length"],
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


def get_arg_parser():
    parser = ai4ts.arma.get_arg_parser()
    parser.add_argument(
        "--order",
        help="comma seprated AR,I,MA orders to use with input file",
    )
    parser.add_argument(
        "-l",
        "--prediction-length",
        type=int,
        required=True,
        help="How many data points to predict using the ARMA model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Pytorch device, e.g. cpu, cuda (nvidia), mps (apple), xpu (intel)",
    )
    return parser


def get_fit_param_arrays(params):
    ar = []
    ma = []
    var = 0.0
    for key in params.index:
        if key.startswith("ar."):
            ar.append(params[key])
        elif key.startswith("ma."):
            ma.append(params[key])
        elif key == "sigma2":
            var = params[key]
    return ar, ma, var


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    order = None
    if args.input_file:
        df = ai4ts.io.read_df(args.input_file)
        if args.order:
            order = args.order.split(",")
        if len(order) != 3:
            parser.error("order must be three comma separated values")
    else:
        if not args.frequency:
            parser.error("frequency is required if input file not specified")

        phi = args.phi if args.phi else []
        theta = args.theta if args.theta else []

        order = (len(phi), 0, len(theta))
        # NB: lag-llama required float32
        df = ai4ts.arma.arma_generate_df(
            args.count,
            phi,
            theta,
            "2024-01-01",
            frequency=args.frequency,
            scale=args.scale_deviation,
            mean=args.mean,
            dtype=np.float32,
        )

    df_train = df.iloc[: -args.prediction_length]

    dataset = PandasDataset(df_train)
    device = torch.device(args.device)

    forecasts, tss = get_lag_llama_predictions(
        dataset=dataset,
        prediction_length=args.prediction_length,
        device=device,
        context_length=32,
        num_samples=100,
    )

    # import ipdb; ipdb.set_trace()

    title = "lag-llama %d point forecast" % args.prediction_length
    if args.input_file:
        title += " (input %s)" % args.input_file
    elif order is not None:
        title += " ARIMA(%d,%d,%d)" % order

    ai4ts.plot.plot_prediction(
        df, forecasts[0].mean, args.output_file, title=title
    )

    """
    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter("%b, %d")
    plt.rcParams.update({"font.size": 15})

    # Iterate through the first 9 series, and plot the predicted samples
    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx + 1)

        plt.plot(
            ts[-4 * args.prediction_length :].to_timestamp(),
            label="target",
        )
        forecast.plot(color="g")
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        ax.set_title(forecast.item_id)

    plt.gcf().tight_layout()
    plt.legend()
    if args.output_file:
        plt.savefig(args.output_file)
    else:
        plt.show()
    """


if __name__ == "__main__":
    main()

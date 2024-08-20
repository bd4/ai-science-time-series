#!/usr/bin/env python3

from itertools import islice
import math
import os.path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from statsmodels.graphics import tsaplots
import statsmodels.tsa.api as tsa

import ai4ts


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

    if args.input_file:
        df = ai4ts.io.read_df(args.input_file)
        if not args.order:
            parser.error("order is required when using input file")
        order = args.order.split(",")
        if len(order) != 3:
            parser.error("order must be three comma separated values")
    else:
        if not args.frequency:
            parser.error("frequency is required if input file not specified")

        phi = args.phi if args.phi else []
        theta = args.theta if args.theta else []

        order = (len(phi), 0, len(theta))
        df = ai4ts.arma.arma_generate_df(
            args.count,
            phi,
            theta,
            "2024-01-01",
            frequency=args.frequency,
            scale=args.scale_deviation,
            mean=args.mean,
        )
        y = df["target"]

    dates = df["ds"]
    df_train = df[-args.prediction_length :]

    model = tsa.ARIMA(df_train["target"], order=order, trend="n")
    model_fit = model.fit()
    fit_ar, fit_ma, fit_var = get_fit_param_arrays(model_fit.params)

    print("Actual: ", phi, theta, args.scale_deviation)
    print("Fit   : ", fit_ar, fit_ma, math.sqrt(fit_var))

    pred = model_fit.get_prediction(start=-args.prediction_length, end=-1)

    # import ipdb; ipdb.set_trace()

    # plot can be done using statsmodels api directly, but this calls prediction under
    # the hood and is not documented well, so it not clear what it is doing. Better
    # to use our own common plot code that is the same for all models
    """
    plot_start_date = dates.iloc[-min(5 * args.prediction_length, len(dates))]
    fig, ax = plt.subplots(figsize=(10, 8))
    df.loc[df["ds"] >= plot_start_date].plot(
        x="ds", y="target", color="lightgreen", ax=ax
    )
    fig = tsaplots.plot_predict(model_fit, start=-args.prediction_length, ax=ax)
    ax.legend(loc="upper left")
    if args.output_file:
        fig.savefig(args.output_file)
    else:
        plt.show()
    """

    forecast_ds = df.ds.iloc[-args.prediction_length :]
    df_forecast = pd.DataFrame({"ds": forecast_ds, "target": pred.predicted_mean})
    ai4ts.plot.plot_prediction(df, df_forecast, args.output_file)


if __name__ == "__main__":
    main()

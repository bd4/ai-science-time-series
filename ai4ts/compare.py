import os.path
from collections import namedtuple
import math

from matplotlib import pyplot as plt
import numpy as np
import yaml

import ai4ts


ARIMAParams = namedtuple("ARIMAParams", "ar i ma")


def float_array_to_str(a, sep=",", fmt="{:1.2f}"):
    return sep.join(fmt.format(val) for val in a)


class ARIMAParams(object):
    def __init__(self, ar, i, ma):
        self.ar = ar
        self.i = i
        self.ma = ma

    @property
    def ar_order(self):
        return len(self.ar)

    @property
    def ma_order(self):
        return len(self.ma)

    @classmethod
    def from_dict(cls, d):
        return cls(d["ar"], d["i"], d["ma"])

    def __str__(self):
        parts = []
        if self.ar:
            parts.append("φ({})".format(float_array_to_str(self.ar)))
        if self.i > 0:
            parts.append("{:d}".format(self.i))
        if self.ma:
            parts.append("θ({})".format(float_array_to_str(self.ma)))
        return ", ".join(parts)


def get_arg_parser():
    parser = ai4ts.arma.get_arg_parser()
    parser.add_argument(
        "-l",
        "--prediction-length",
        type=int,
        required=True,
        help="How many data points to predict using the ARMA model",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=int,
        default=3,
        help="How many columns to use for plot grid",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Pytorch device, e.g. cpu, cuda (nvidia), mps (apple), xpu (intel)",
    )
    return parser


def read_arma_test_yaml(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return [ARIMAParams.from_dict(case) for case in data["test_cases"]]


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    params = []
    if args.input_file:
        _, ext = os.path.splitext(args.input_file)
        if ext == ".yaml":
            params = read_arma_test_yaml(args.input_file)

    if not args.frequency:
        parser.error("-f/--frequency argument is required")
    if not args.count:
        parser.error("-n/--count argument is required")

    if args.phi or args.theta:
        params.append(ARIMAParams(args.phi, 0, args.theta))

    forecast_fns = [
        ai4ts.arma.forecast,
        ai4ts.lag_llama.forecast,
        ai4ts.chronos.forecast,
        ai4ts.timesfm.forecast,
    ]
    ncols = 3
    nrows = math.ceil(len(params) / ncols)

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex="col", layout="constrained"
    )
    fig.autofmt_xdate(rotation=60)

    # Note: lag-llama requires float32 input data
    dtype = np.float32
    for i, p in enumerate(params):
        df = ai4ts.arma.arma_generate_df(
            args.count,
            p.ar,
            p.ma,
            start_date="2024-01-01",
            frequency=args.frequency,
            scale=args.scale_deviation,
            mean=args.mean,
            dtype=dtype,
        )
        df_train = df.iloc[: -args.prediction_length]

        irow = i // ncols
        icol = i % ncols

        ax = axs[irow, icol]

        forecast_map = {}
        for forecast_fn in forecast_fns:
            fcast = forecast_fn(
                df_train["ds"],
                df_train["target"],
                args.prediction_length,
                p.ar_order,
                p.i,
                p.ma_order,
                device=args.device,
            )
            forecast_map[fcast.name] = fcast.data
        ai4ts.plot.plot_prediction(df, forecast_map, ax=ax, legend=False, title=str(p))

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")

    if args.output_file:
        fig.savefig(args.output_file)
    else:
        plt.show()

import os.path
from collections import namedtuple
import math

from matplotlib import pyplot as plt
import yaml

import ai4ts


ARIMAParams = namedtuple("ARIMAParams", "ar i ma")


def float_array_to_str(a, precision=2, sep=",", fmt="{:1.2f}"):
    return "[{}]".format(sep.join(fmt.format(val) for val in a))


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
        return "ARIMA({0}, {1}, {2})".format(
            float_array_to_str(self.ar), self.i, float_array_to_str(self.ma)
        )


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

    ncols = args.columns
    nparams = len(params)
    nrows = math.ceil(nparams / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex="col")
    fig.autofmt_xdate(rotation=60)

    for x in range(ncols):
        for y in range(nrows):
            ax = axs[y, x]
            idx = y * nrows + x
            if idx >= nparams:
                break
            p = params[idx]
            df = ai4ts.arma.arma_generate_df(
                args.count,
                p.ar,
                p.ma,
                start_date="2024-01-01",
                frequency=args.frequency,
                scale=args.scale_deviation,
                mean=args.mean,
            )
            df_train = df.iloc[: -args.prediction_length]
            _, fcast = ai4ts.arma.forecast(
                df_train["target"], args.prediction_length, p.ar_order, p.i, p.ma_order
            )
            ai4ts.plot.plot_prediction(df, fcast, ax=ax, title=str(p))

    if args.output_file:
        fig.savefig(args.output_file)
    else:
        plt.show()

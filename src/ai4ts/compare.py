import os.path
import math
import importlib

from matplotlib import pyplot as plt
import numpy as np
import yaml

import ai4ts
import ai4ts.arma


MODEL_NAME_CLASS = {
    "arma": "ai4ts.arma.ARIMAModel",
    "auto-arma": "ai4ts.arma.AutoARIMAModel",
    "lag-llama": "ai4ts.lag_llama.LagLlamaModel",
    "chronos": "ai4ts.chronos.ChronosModel",
    "timesfm": "ai4ts.timesfm.TimesFmModel",
}


def _get_model_class(dotted_name):
    module_name, class_name = dotted_name.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def float_array_to_str(a, sep=",", fmt="{:1.2f}"):
    return sep.join(fmt.format(val) for val in a)


class ARIMAParams(object):
    def __init__(self, ar, coeff, ma):
        self.ar = ar
        self.coeff = coeff
        self.ma = ma

    @property
    def ar_order(self):
        return len(self.ar)

    @property
    def ma_order(self):
        return len(self.ma)

    @property
    def i(self):
        return max(0, len(self.coeff) - 1)

    @classmethod
    def from_dict(cls, d):
        return cls(d["ar"], d["coeff"], d["ma"])

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
    parser.add_argument(
        "--trend-poly-coeff",
        dest="coeff",
        nargs="*",
        type=float,
        help="List of coefficients for trend polynomial (constant first)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODEL_NAME_CLASS.keys()),
        default=list(MODEL_NAME_CLASS.keys()),
        help="List models to compare (default all)",
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
        params.append(ARIMAParams(args.phi, args.coeff, args.theta))

    ncols = 3
    nrows = math.ceil(len(params) / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex="col", layout="tight")
    fig.autofmt_xdate(rotation=60)

    trend_fig = plt.figure()
    trend_axs = trend_fig.subplots(nrows=nrows, ncols=ncols)

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

        irow = i // ncols
        icol = i % ncols
        ax = axs[irow, icol]

        if p.coeff:
            trend = ai4ts.arma.get_trend(len(df.target), p.coeff, dtype=dtype)
            trend_ax = trend_axs[irow, icol]
            trend_ax.plot(trend)
            trend_ax.set_title(",".join(str(x) for x in p.coeff))
            df.target += trend
        df_train = df.iloc[: -args.prediction_length]

        model_class_names = [MODEL_NAME_CLASS[n] for n in args.models]

        forecast_map = {}
        for class_name in model_class_names:
            mclass = _get_model_class(class_name)
            m = mclass()
            m.fit(
                df_train["ds"],
                df_train["target"],
                args.prediction_length,
                ar_order=p.ar_order,
                i_order=p.i,
                ma_order=p.ma_order,
                device=args.device,
            )
            fcast = m.predict(args.prediction_length)
            forecast_map[fcast.name] = fcast.data
        # import ipdb; ipdb.set_trace()
        ai4ts.plot.plot_prediction(df, forecast_map, ax=ax, legend=False, title=str(p))

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")

    if args.output_file:
        fig.savefig(args.output_file)
    else:
        plt.show()

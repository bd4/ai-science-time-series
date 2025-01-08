import os.path
import math
import importlib

from matplotlib import pyplot as plt
import numpy as np
import yaml

import ai4ts
import ai4ts.arma
from ai4ts.arma import ARIMAParams


MODEL_NAME_CLASS = {
    "arma": "ai4ts.arma.ARIMAModel",
    "auto-arma": "ai4ts.arma.AutoARIMAModel",
    "lag-llama": "ai4ts.lag_llama.LagLlamaModel",
    "chronos": "ai4ts.chronos.ChronosModel",
    "timesfm": "ai4ts.timesfm.TimesFmModel",
    "simplelinear": "ai4ts.neuralnet.SimpleLinearModel",
    "simplenn": "ai4ts.neuralnet.SimpleNNModel",
    "rnn": "ai4ts.neuralnet.RNNModel",
}


def mean_absolute_error(a, b):
    """
    Hack to make work with numpy and/or torch
    """
    if isinstance(a, np.ndarray):
        return np.mean(np.absolute(a - b))
    else:
        import torch

        a = torch.tensor(a)
        return torch.mean(torch.abs(a - b))


def _get_model_class(dotted_name):
    module_name, class_name = dotted_name.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


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
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=list(ai4ts.data.DATASETS.keys()),
        default=list(ai4ts.data.DATASETS.keys()),
        help="List datasets to fit (default all)",
    )

    return parser


def read_arma_test_yaml(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return [ARIMAParams.from_dict(case) for case in data["test_cases"]]


def plot_dataset_predictions(
    model_class_names,
    datasets,
    prediction_length,
    device="cpu",
    output_file=None,
    ncols=3,
):
    nrows = math.ceil(len(datasets) / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex="col", layout="tight")
    fig.autofmt_xdate(rotation=60)

    for i, dataset in enumerate(datasets):
        if nrows > 1:
            irow = i // ncols
            icol = i % ncols
            ax = axs[irow, icol]
        else:
            ax = axs[i]

        df = dataset.get_transformed_data()
        df_train = df.iloc[:-prediction_length]

        print(f"=== {dataset.get_description()} ===")
        forecast_map = {}
        for class_name in model_class_names:
            mclass = _get_model_class(class_name)
            m = mclass()
            m.fit(
                df_train["ds"],
                df_train["target"],
                prediction_length,
                ar_order=dataset.ar_order,
                i_order=dataset.i,
                ma_order=dataset.ma_order,
                device=device,
            )
            fcast = m.predict(prediction_length)
            forecast_map[fcast.name] = fcast.data
            mae = mean_absolute_error(
                fcast.data, df.iloc[-prediction_length:]["target"].values
            )
            print(f"{fcast.name:15s}{mae:.3f}\t{str(fcast.model)}")
        print()
        # import ipdb; ipdb.set_trace()
        ai4ts.plot.plot_prediction(
            df, forecast_map, ax=ax, legend=False, title=dataset.get_description()
        )

    if nrows > 1:
        handles, labels = axs[0, 0].get_legend_handles_labels()
    else:
        handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")

    if output_file:
        fig.savefig(output_file)
    else:
        plt.show()


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

    # trend_fig = plt.figure()
    # trend_axs = trend_fig.subplots(nrows=nrows, ncols=ncols)

    ar_datasets = []
    real_datasets = []

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

        if p.coeff:
            trend = ai4ts.arma.get_trend(len(df.target), p.coeff, dtype=dtype)
            df.target += trend

        ar_datasets.append(ai4ts.data.SimpleTimeseries(df, str(p)))

    for ds_name in args.datasets:
        real_datasets.append(ai4ts.data.DATASETS[ds_name])

    model_class_names = [MODEL_NAME_CLASS[n] for n in args.models]

    plot_dataset_predictions(
        model_class_names,
        ar_datasets,
        args.prediction_length,
        device=args.device,
        output_file=args.output_file,
        ncols=3,
    )
    # plot_dataset_predictions(model_class_names, real_datasets)

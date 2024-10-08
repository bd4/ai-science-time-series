#!/usr/bin/env python3

import math

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

    df_train = df.iloc[: -args.prediction_length]

    model_fit, forecast = ai4ts.arma.forecast(
        df_train["target"],
        prediction_length=args.prediction_length,
        order=order,
    )

    fit_ar, fit_ma, fit_var = get_fit_param_arrays(model_fit.params)

    print("Actual: ", phi, theta, args.scale_deviation)
    print("Fit   : ", fit_ar, fit_ma, math.sqrt(fit_var))

    # import ipdb; ipdb.set_trace()

    ai4ts.plot.plot_prediction(
        df,
        forecast,
        args.output_file,
    )


if __name__ == "__main__":
    main()

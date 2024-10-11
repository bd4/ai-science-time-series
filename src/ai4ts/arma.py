import os.path

import numpy as np
import pandas as pd

import statsmodels.api as sm

import ai4ts


SEED = 42


def main():
    np.random.seed(SEED)

    parser = get_arg_parser()
    args = parser.parse_args()

    if args.input_path:
        df = ai4ts.io.read_df(args.input_path, parser=parser)
    else:
        if not args.phi and not args.theta:
            parser.error("phi or theta are required when input path is not specified")
        elif args.frequency:
            df = arma_generate_df(
                args.count,
                args.phi,
                args.theta,
                start_date="2024-01-01",
                frequency=args.frequency,
                scale=args.scale_deviation,
                mean=args.mean,
            )
            y = df["target"]
        else:
            y = arma_generate(
                args.count,
                args.phi,
                args.theta,
                scale=args.scale_deviation,
                mean=args.mean,
            )
            df = None

    if args.output_file:
        if args.frequency:
            ai4ts.io.write_df(df, args.output_file)
        else:
            _, ext = os.path.splitext(args.output_file)
            if ext == ".txt" or ext == "":
                np.savetxt(args.output_file, y)
            elif ext == ".npy":
                np.save(args.output_file, y)
            else:
                parser.error("Unknown output format for numpy: '%s'" % ext)
    else:
        if args.frequency:
            print(df)
        else:
            print(y[:10])
            if len(y) > 20:
                print("...")
                print(y[-10:])


def arma_generate(n, phi, theta, scale=1.0, mean=0, frequency=None, dtype=None):
    """
    Generate a sequence with an ARMA process

    :param n: length of sequence to generate
    :param phi: list of AR coefficients
    :param theta: list of MA coefficients
    :param scale: standard deviation of noise
    :param mean: mean of generated data (default 0.0)

    :returns: numpy array of n elements
    """
    # Note: statsmodes expects lag operator coefficients, including the 0th. In
    # practice this means we need to pass 1 and negate the ar params.
    arparams = np.r_[1, -np.array(phi)]
    maparams = np.r_[1, np.array(theta)]
    y = sm.tsa.arma_generate_sample(arparams, maparams, nsample=n, scale=scale)
    y += mean

    if dtype is not None:
        y = y.astype(dtype)

    return y


def get_trend(n, coeff, dtype=None):
    """
    Create a polynomial trend array with specified coefficients.

    :param n: length of trend array
    :param coeff: polynomial coefficients, starting with constant term increasing
                  powers to the right
    :param dtype: optional numpy dtype
    """

    coeff = np.array(coeff, dtype=dtype)

    def trend_fn(x):
        y = coeff[0]
        for p in range(1, len(coeff)):
            y += coeff[p] * pow(x, p)
        return y

    trend_fn_vec = np.vectorize(trend_fn, otypes=[dtype])
    return trend_fn_vec(np.arange(n, dtype=dtype))


def arma_generate_df(
    n,
    phi,
    theta,
    start_date,
    frequency,
    scale=1.0,
    mean=0,
    dtype=None,
    time_column="ds",
    data_column="target",
):
    """
    Generate a sequence with an ARMA process

    :param n: length of sequence to generate
    :param phi: list of AR coefficients
    :param theta: list of MA coefficients
    :param start_date: date string to start time column from
    :param frequency: string code for time frequency (s, m, h, d, w, m, y, etc)
    :param scale: standard deviation of noise
    :param mean: mean of generated data (default 0.0)
    :param dtype: numpy datatype to use
    :param time_column: name of column for date sequence
    :param data_column: name of column for generated data points

    :returns: pandas dataframe with time and data columns named according to
      the specified arguments, and a 'frequency' column with the specified
      frequency code
    """
    y = arma_generate(n, phi, theta, scale=scale, mean=mean, dtype=dtype)
    dates = pd.date_range(start_date, freq=frequency.lower(), periods=n)
    df = pd.DataFrame({time_column: dates, data_column: y})
    df["frequency"] = frequency
    return df


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="generate ARMA dataframe")
    parser.add_argument("-p", "--phi", nargs="+", type=float, help="AR coefficients")
    parser.add_argument("-t", "--theta", nargs="+", type=float, help="MA coefficients")
    parser.add_argument(
        "-u",
        "--mean",
        type=float,
        default=0,
        help="mean of series to generate (default 0)",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1000,
        help="number of elements to generate (default 1000)",
    )
    parser.add_argument(
        "-s",
        "--scale-deviation",
        type=float,
        default=1,
        help="white noise standard deviation (default 1)",
    )
    parser.add_argument("-o", "--output-file", required=False)
    parser.add_argument(
        "-i",
        "--input-file",
        help="read DataFrame from file instead of generating ARMA data",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        help="time frequency for generated data (s, h, d, m, y)",
    )
    return parser


class ARIMAModel(ai4ts.model.TimeSeriesModel):
    def fit(self, times, history, max_prediction_length, **kwargs):
        import statsforecast.models as sfmodels

        if isinstance(history, pd.Series):
            history = history.values
        self.order = kwargs.get("order")
        if self.order is None:
            ar_order = kwargs.get("ar_order", 0)
            i_order = kwargs.get("i_order", 0)
            ma_order = kwargs.get("ma_order", 0)
            self.order = (ar_order, i_order, ma_order)
        self.model = sfmodels.ARIMA(order=self.order)
        self.model.fit(history)

    def predict(self, prediction_length):
        prediction = self.model.predict(prediction_length)
        return ai4ts.model.Forecast(prediction["mean"], "ARIMA CSS-ML", self.model)


class AutoARIMAModel(ai4ts.model.TimeSeriesModel):
    def fit(self, times, history, max_prediction_length, **kwargs):
        import statsforecast.models as sfmodels

        if isinstance(history, pd.Series):
            history = history.values
        self.model = sfmodels.AutoARIMA(seasonal=False)
        self.model.fit(history)

    def predict(self, prediction_length):
        prediction = self.model.predict(prediction_length)
        return ai4ts.model.Forecast(prediction["mean"], "Auto-ARIMA", self.model)


if __name__ == "__main__":
    main()

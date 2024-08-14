import os.path

import numpy as np
import pandas as pd
import statsmodels.api as sm


SEED = 42

from . import io


def main():
    np.random.seed(SEED)

    parser = get_arg_parser()
    args = parser.parse_args()

    if args.input_path:
        df = io.read_df(args.input_path, parser=parser)
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
            io.write_df(df, args.output_file)
        else:
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


def arma_generate(n, phi, theta, scale=1.0, mean=0, frequency=None):
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

    return y


def arma_generate_df(n, phi, theta, start_date, frequency, scale=1.0, mean=0):
    """
    Generate a sequence with an ARMA process

    :param n: length of sequence to generate
    :param phi: list of AR coefficients
    :param theta: list of MA coefficients
    :param start_date: date string to start time column from
    :param frequency: string code for time frequency (s, m, h, d, w, m, y, etc)
    :param scale: standard deviation of noise
    :param mean: mean of generated data (default 0.0)

    :returns: pandas dataframe with columns 'ds' (dates), 'target" (generated
      arma sequence), 'frequency' (copy of specified frequency code)
                                                                    d
    """
    y = arma_generate(n, phi, theta, scale=scale, mean=mean)
    dates = pd.date_range(start_date, freq=frequency, periods=n)
    # Note: use column names that model libraries expect. Some libraries
    # may still require renaming to work.
    df = pd.DataFrame({"ds": dates, "target": y}, pd.Index(dates, name="ds"))
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


if __name__ == "__main__":
    main()

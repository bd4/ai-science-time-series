#!/usr/bin/env python3

import os.path

import numpy as np
import pandas as pd
import statsmodels.api as sm


SEED = 42


def main():
    np.random.seed(SEED)

    parser = get_arg_parser()
    args = parser.parse_args()

    y = arma_generate(
        args.count, args.phi, args.theta, scale=args.scale_deviation, mean=args.mean
    )

    df = None
    if args.frequency:
        start_date = "2024-01-01"
        dates = pd.date_range(start_date, freq=args.frequency, periods=args.count)
        # Note: use column names that model libraries expect. Some libraries
        # may still require renaming to work.
        df = pd.DataFrame({"target": y, "ds": dates})
        df["frequency"] = args.frequency

    if args.output_file:
        if args.frequency:
            _, ext = os.path.splitext(args.output_file)
            if ext == ".csv":
                df.to_csv(args.output_file)
            elif ext == ".parquet":
                df.to_parquet(args.output_file)
            elif ext == ".hdf":
                df.to_hdf(args.output_file)
            elif ext == ".pickle":
                df.to_pickle(args.output_file)
            else:
                parser.error("Unknown output format for pandas: '%s'" % ext)
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


def arma_generate(n, phi, theta, scale=1.0, mean=0):
    # Note: statsmodes expects lag operator coefficients, including the 0th. In
    # practice this means we need to pass 1 and negate the ar params.
    arparams = np.r_[1, -np.array(phi)]
    maparams = np.r_[1, np.array(theta)]
    y = sm.tsa.arma_generate_sample(arparams, maparams, nsample=n, scale=scale)
    y += mean
    return y


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="generate ARMA dataframe")
    parser.add_argument(
        "-p", "--phi", nargs="+", type=float, required=True, help="AR coefficients"
    )
    parser.add_argument(
        "-t", "--theta", nargs="+", type=float, required=True, help="MA coefficients"
    )
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
        "-f",
        "--frequency",
        help="output a dataframe with time column of specified frequency (H, D, M, Y)",
    )
    return parser


if __name__ == "__main__":
    main()

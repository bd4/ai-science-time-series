#!/usr/bin/env python3

import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa


SEED = 42

def main():
    np.random.seed(SEED)

    args = get_args()

    y = arma_generate(args.count, args.phi, args.theta,
                      scale=args.scale_deviation, mean=args.mean)

    if args.output_file:
        np.savetxt(args.output_file, y)
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
    y = tsa.arma_generate_sample(arparams, maparams,
                                 nsample=n, scale=scale)
    return y
    #y += mean * (1 - np.sum(phi) - np.sum(theta))

    dates = pd.date_range(start_date, freq=frequency, periods=n)
    return pd.DataFrame(index=dates, data=y, columns=["value"])


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
                description="generate ARMA dataframe")
    parser.add_argument("-p", "--phi", nargs="+", type=float, required=True,
                        help="AR coefficients")
    parser.add_argument("-t", "--theta", nargs="+", type=float, required=True,
                        help="MA coefficients")
    parser.add_argument("-u", "--mean", type=float, default=0,
                        help="mean of series to generate (default 0)")
    parser.add_argument("-n", "--count", type=int, default=1000,
                        help="number of elements to generate (default 1000)")
    parser.add_argument("-s", "--scale-deviation", type=float, default=1,
                        help="white noise standard deviation (default 1)")
    parser.add_argument("-o", "--output-file", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()

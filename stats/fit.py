#!/usr/bin/env python3

from itertools import islice
import os.path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from statsmodels.graphics import tsaplots
import statsmodels.tsa.api as tsa


def main(inpath, outpath, prediction_length=24):
    values = np.loadtxt(inpath, dtype=np.float32)
    dates = pd.date_range("2024-01-01", periods=len(values), freq="1H")
    df = pd.Series(values, index=dates)
    df_train = df[-prediction_length:]

    n_cycles = len(df) / prediction_length

    model = tsa.ARIMA(df_train, order=(2, 0, 2), trend="n")
    forecast = model.fit()

    plot_start_date = "2024-04-03"
    predict_start_date = dates[-prediction_length]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df.loc[plot_start_date:], color="lightgreen")
    fig = tsaplots.plot_predict(forecast, start=predict_start_date, ax=ax)
    legend = ax.legend(loc="upper left")
    fig.savefig(outpath)


if __name__ == "__main__":
    import sys

    inpath = sys.argv[1]
    outpath, _ = os.path.splitext(inpath)
    outpath += "-statsmodels.pdf"
    main(inpath, outpath)
    print(outpath)

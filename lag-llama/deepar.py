#!/usr/bin/env python3

import os.path

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator

def main(inpath, outpath):
    # Load data from a CSV file into a PandasDataset
    if inpath is None:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/AileenNielsen/"
            "TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
            index_col=0,
            parse_dates=True,
        )
        dataset = PandasDataset(df, target="#Passengers")
        freq = "M"
    else:
        values = np.loadtxt(inpath, dtype=np.float32)
        df = pd.DataFrame({ "target": values },
                          index=pd.date_range("2024-01-01", periods=len(values), freq="1H"))
        dataset = PandasDataset(df)
        freq = "H"

    # Split the data for training and testing
    training_data, test_gen = split(dataset, offset=-36)
    test_data = test_gen.generate_instances(prediction_length=12, windows=3)

    # Train the model and make predictions
    model = DeepAREstimator(
        prediction_length=12, freq=freq, trainer_kwargs={"max_epochs": 5}
    ).train(training_data)

    forecasts = list(model.predict(test_data.input))

    # Plot predictions
    plt.plot(df, color="black")
    for forecast in forecasts:
      forecast.plot()
    plt.legend(["True values"], loc="upper left", fontsize="xx-large")
    #plt.show()
    plt.savefig(outpath)


if __name__ == "__main__":
    import sys
    inpath = sys.argv[1]
    outpath, _ = os.path.splitext(inpath)
    outpath += "_deepar.pdf"
    main(inpath, outpath)
    print(outpath)

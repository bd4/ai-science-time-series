#!/usr/bin/env python3

import os.path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import timesfm


def main(inpath, outpath, prediction_length=24, backend="gpu"):
    values = np.loadtxt(inpath, dtype=np.float32)
    dates = pd.date_range("2024-01-01", periods=len(values), freq="1H")
    df = pd.DataFrame({"y": values, "ds": dates})
    df["unique_id"] = 0
    print("cols", df.columns)
    df_train = df[:-prediction_length]

    tfm = timesfm.TimesFm(
        context_len=512,
        horizon_len=prediction_length * 4,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend=backend,
    )
    # train_state_unpadded_shape_dtype_struct
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    df_future = tfm.forecast_on_df(
        inputs=df_train, freq="H", value_name="y", num_jobs=-1
    )

    history_tail = df[-min(4 * prediction_length, len(values)) :]

    plt.figure(figsize=(8, 4))
    plt.plot(history_tail, color="royalblue", label="historical data")
    plt.plot(df_future, color="tomato", label="forecast")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(outpath)


if __name__ == "__main__":
    import sys

    inpath = sys.argv[1]
    outpath, _ = os.path.splitext(inpath)
    outpath += "-timesfm.pdf"
    main(inpath, outpath)
    print(outpath)

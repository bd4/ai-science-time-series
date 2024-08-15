import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_prediction(
    df_actual, df_prediction, outpath=None, n_cycles=5, x_column="ds", y_column="target"
):
    """
    Plot data and predictions.

    :param df_actual: DataFrame with historical data + actual data for prediction period
    :param df_prediction: Predicted data based on historical data
    :param outpath: save to this path instead of displaying
    :param n_cycles: number of prediction length cycles to include in the plot
    """
    prediction_length = len(df_prediction)
    display_length = n_cycles * prediction_length
    df_actual_display = df_actual[-min(display_length, len(df_actual)) :]

    fig, ax = plt.subplots(figsize=(8, 4))
    df_actual_display.plot(
        x=x_column, y=y_column, ax=ax, color="royalblue", label="actual data"
    )
    df_prediction.plot(x=x_column, y=y_column, ax=ax, color="tomato", label="forecast")
    ax.legend()
    if outpath:
        fig.savefig(outpath)
    else:
        plt.show()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_prediction(
    df_actual,
    prediction_array,
    outpath=None,
    display_cycles=5,
    time_column="ds",
    data_column="target",
):
    """
    Plot data and predictions.

    :param df_actual: DataFrame with historical data + actual data for prediction period
    :param prediction_array: np.array or pd.Series of predicted data
    :param outpath: save to this path instead of displaying
    :param display_cycles: number of prediction length cycles to include in the plot
    """
    prediction_length = len(prediction_array)
    display_length = display_cycles * prediction_length
    df_actual_display = df_actual[-min(display_length, len(df_actual)) :]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        df_actual_display[time_column],
        df_actual_display[data_column],
        color="royalblue",
        label="actual data",
    )
    ax.plot(
        df_actual[time_column].iloc[-prediction_length:],
        prediction_array,
        color="tomato",
        label="forecast",
    )
    ax.legend()
    if outpath:
        fig.savefig(outpath)
    else:
        plt.show()

from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def plot_prediction(
    df_actual,
    prediction_array,
    outpath=None,
    display_cycles=5,
    time_column="ds",
    data_column="target",
    title=None,
    ax=None
):
    """
    Plot data and predictions. If ax was passed, plots on specified axis and does
    not display. If ax not passed, save figure to output path if specified, or
    show if not.

    :param df_actual: DataFrame with historical data + actual data for prediction period
    :param prediction_array: np.array or pd.Series of predicted data
    :param outpath: save to this path instead of displaying
    :param display_cycles: number of prediction length cycles to include in the plot
    :param time_column: name of column with time data
    :param data_column: name of column with series data
    :param title: set axis title (optional)
    :param ax: plot to passed matplotlib axis and do not show or display
    """
    prediction_length = len(prediction_array)
    display_length = display_cycles * prediction_length
    df_actual_display = df_actual[-min(display_length, len(df_actual)) :]

    date_formater = mdates.DateFormatter("%b, %d")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.autofmt_xdate(rotation=60)

    ax.xaxis.set_major_formatter(date_formater)
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

    if title is not None:
        ax.set_title(title)

    if ax is None:
        if outpath:
            fig.savefig(outpath)
        else:
            plt.show()

import os.path
import abc

import numpy as np
import pandas as pd


def create_lag_input_output(series_data, lag):
    """
    Set up arrays for training an autoregressive neural network using
    the specified lag. Lag 1 correspends to an AR(1), etc.
    """
    series_len = len(series_data)
    in_arrays = [torch.tensor(series_data[i - lag : i]) for i in range(lag, series_len)]
    in_matrix = torch.stack(in_arrays, 0)
    out_vector = torch.tensor([series_data[i] for i in range(lag, series_len)])
    return (in_matrix, out_vector)


class TimeseriesDataset(abc.ABC):
    @abc.abstractmethod
    def get_transformed_data(self):
        """
        Get pandas dataframe of data that has been processed to make fitting and
        prediction easier, e.g. taking log and/or differencing.
        """
        ...

    @abc.abstractmethod
    def get_actual_data(self):
        """
        Get actual data with all na's removed.
        """
        ...

    @abc.abstractmethod
    def prediction_to_actual_data(self, prediction_df):
        """
        Given predictions based on transformed data, convert to the original
        data format, i.e. reverse differencing and any functionas applied.
        """
        ...

    @abc.abstractmethod
    def get_description(self):
        """Return short description string suitable for use in legends"""
        ...

    def get_lag_input_output_data(self, lag):
        return create_lag_input_output(self.get_transformed_data(), lag)

    # hack to emulate compare.ARIMAParams
    @property
    def ar_order(self):
        return 1

    @property
    def ma_order(self):
        return 1

    @property
    def i(self):
        # Note: differencing is handled by transformations, so models
        # only see the differenced data
        return 0


class SimpleTimeseries(TimeseriesDataset):
    def __init__(self, df, description):
        self.df = df
        self.description = description

    def get_transformed_data(self):
        return self.df

    def get_actual_data(self):
        return self.df

    def prediction_to_actual_data(self, prediction_series):
        return prediction_series

    def get_description(self):
        return self.description


class GlacialVarve(TimeseriesDataset):
    def __init__(self):
        import astsadata

        # 4th edition Example 2.7, says data start "11,834 years ago", publish date 2017

        self.df_actual = pd.read_csv(
            f"{astsadata.path}/data/varve.csv",
            header=0,
            names=["index", "target"],
            index_col=0,
            parse_dates=False,
            dtype={"index": np.int32, "target": np.float32},
        )
        # hack for models that want dates, and just start at 1000 CE, actual
        # dates don't really matter for fitting
        self.df_actual["ds"] = pd.date_range(
            "1000", freq="Y", periods=len(self.df_actual), unit="s"
        )
        self.prediction_start = self.df_actual.index[-1]
        self.df_difflog = None

    def get_actual_data(self):
        return self.df_actual

    def get_transformed_data(self):
        # Note: Shumway example 3.33, this fits MA(1) reasonably well
        if self.df_difflog is None:
            self.df_difflog = self.df_actual.copy()
            self.df_difflog["target"] = (
                self.df_difflog["target"].map(np.log).diff().astype(np.float32)
            )
            # remove first row with NA diff
            self.df_difflog = self.df_difflog.iloc[1:]
        return self.df_difflog

    def prediction_to_actual_data(self, df_predicted):
        last = self.df_actual[-1]
        log_actual = np.cumsum(df_predicted["target"].values) + last
        log_actual["target"] = log_actual["target"].map(np.exp)
        return log_actual

    def get_description(self):
        return "glacial-varve"

    @property
    def ar_order(self):
        return 0

    @property
    def ma_order(self):
        return 1

    def __str__(self):
        return self.get_description()


DATASETS = dict(
    glacial_varve=GlacialVarve(),
)

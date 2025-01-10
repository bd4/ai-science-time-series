import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ai4ts
import ai4ts.model
import ai4ts.arma
import ai4ts.lag_llama
import ai4ts.chronos


class NaiveModel(ai4ts.model.TimeSeriesModel):
    """
    Model that predicts the last known value in history as
    all future values.
    """

    deep = False

    def fit(self, times, history, max_prediction_length, **kwargs):
        if isinstance(history, pd.Series):
            history = history.values
        self.last = history[-1]

    def predict(self, prediction_length):
        return np.full(prediction_length, self.last)


MODEL_NAME_CLASS = {
    "naive": NaiveModel,
    "arma": ai4ts.arma.ARIMAModel,
    #"auto-arma": "ai4ts.arma.AutoARIMAModel",
    "lag-llama": ai4ts.lag_llama.LagLlamaModel,
    "chronos": ai4ts.chronos.ChronosModel,
    #"timesfm": "ai4ts.timesfm.TimesFmModel",
}


def main():
    print("============ ar(1)[0.9) =============")
    compare_arma_one_ahead_predictors([0.9], [])
    print("============ ar(1)[0.5) =============")
    compare_arma_one_ahead_predictors([0.5], [])
    print("============ ar(1)[0.1] =============")
    compare_arma_one_ahead_predictors([0.1], [])


def compare_arma_one_ahead_predictors(ar, ma):
    df = ai4ts.arma.arma_generate_df(
       1000,
       ar,
       ma,
       start_date="2024-01-01",
       frequency="H",
       scale=1,
       mean=0,
       dtype=np.float32,
    )

    # plt.plot(df["ds"], df["target"], label="true data")

    n_base = 500
    n_predict = 500

    n_models = len(MODEL_NAME_CLASS)
    
    sum_sq_err = np.zeros(n_models)

    prediction_start_date = df["ds"].iloc[n_base]
    predictions = [np.zeros(n_predict) for i in range(n_models)]

    for i in range(n_predict):
        df_train = df.iloc[:n_base + i]
        actual_point = df["target"].iloc[n_base + i]

        for j, mclass in enumerate(MODEL_NAME_CLASS.values()):
            m = mclass()
            m.fit(
                df_train["ds"],
                df_train["target"],
                1,
                ar_order=1
            )

            fcast = m.predict(1)
            next_point = fcast.data[0]
            predictions[j][i] = next_point
            sum_sq_err[j] += (next_point - actual_point)**2

    prediction_dates = pd.date_range(prediction_start_date, freq="H", periods=n_predict)
    for i, name in enumerate(MODEL_NAME_CLASS.keys()):
        plt.figure()
        plt.plot(df["ds"], df["target"], label="true data")
        plt.plot(prediction_dates, predictions[i], label=name)
        plt.legend()
        plt.show()

    for i, name in enumerate(MODEL_NAME_CLASS.keys()):
        rmse = math.sqrt(sum_sq_err[i])
        print(f"{name:15s} rmse: {rmse}")


if __name__ == "__main__":
    main()

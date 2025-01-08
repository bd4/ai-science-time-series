import pandas as pd

import timesfm

import ai4ts


class TimesFmModel(ai4ts.model.TimeSeriesModel):
    # TODO: pass freq to all forecast routines
    def fit(self, times, history, max_prediction_length, **kwargs):
        device = kwargs.get("device", "cuda")
        self.freq = kwargs.get("freq", "H")
        # Uses it's own backend names "gpu", "cpu", or "tpu"
        if device in ["cuda", "xla"]:
            device = "gpu"
        elif device != "cpu":
            device = "tpu"

        if isinstance(history, pd.Series):
            history = history.values

        self.history = pd.DataFrame({"y": history, "ds": times})
        self.history["unique_id"] = 0

        self.model = timesfm.TimesFm(
            context_len=512,
            horizon_len=max_prediction_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=device,
        )
        # train_state_unpadded_shape_dtype_struct
        self.model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    def predict(self, prediction_length):
        # TODO: pass frequency
        df_future = self.model.forecast_on_df(
            inputs=self.history, freq=self.freq, value_name="y", num_jobs=-1
        )

        return ai4ts.model.Forecast(df_future.timesfm, "TimesFM", self.model)

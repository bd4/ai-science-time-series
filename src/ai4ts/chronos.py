#!/usr/bin/env python3

import ai4ts


class ChronosModel(ai4ts.model.TimeSeriesModel):
    def fit(self, times, history, max_prediction_length, **kwargs):
        import torch
        from chronos import ChronosPipeline

        device = kwargs.get("device", "cuda")
        self.model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.history = torch.tensor(history)

    def predict(self, prediction_length):
        # context must be either a 1D tensor, a list of 1D tensors,
        # or a left-padded 2D tensor with batch as the first dimension
        # forecast shape: [num_series, num_samples, prediction_length]
        forecast = self.model.predict(
            context=self.history,
            prediction_length=prediction_length,
            num_samples=100,
        )

        forecast_mean = forecast[0].mean(0)

        return ai4ts.model.Forecast(forecast_mean, "chronos", forecast)

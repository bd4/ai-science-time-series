import pandas as pd

import ai4ts


class LagLlamaModel(ai4ts.model.TimeSeriesModel):
    def fit(self, times, history, max_prediction_length, **kwargs):
        import torch
        from gluonts.dataset.pandas import PandasDataset
        from lag_llama.gluon.estimator import LagLlamaEstimator

        self.num_samples = kwargs.get("num_samples", 100)
        device = kwargs.get("device", "cuda")

        # Note: lag-llama requires times as index
        df = pd.DataFrame({"target": history}, index=times)
        self.history = PandasDataset(df)
        device = torch.device(device)

        ckpt_path = ai4ts.get_model_path("lag-llama.ckpt")
        ckpt = torch.load(ckpt_path, map_location=device)

        context_length = kwargs.get("context_length", 32)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=max_prediction_length,
            # Lag-Llama was trained with a context length of 32,
            # but can work with any context length
            context_length=context_length,
            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            rope_scaling=None,
            batch_size=1,
            num_parallel_samples=100,
            device=device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        self.model = estimator.create_predictor(transformation, lightning_module)

    def predict(self, prediction_length):
        from gluonts.evaluation import make_evaluation_predictions

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.history, predictor=self.model, num_samples=self.num_samples
        )
        forecasts = list(forecast_it)

        return ai4ts.model.Forecast(
            forecasts[0].mean, name="lag-llama", model=forecasts[0]
        )

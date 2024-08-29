import pandas as pd

import timesfm

import ai4ts


# TODO: pass freq to all forecast routines
def forecast(
    times,
    history,
    prediction_length,
    ar_order,
    i_order,
    ma_order,
    device="cuda",
    freq="H",
):
    # Uses it's own backend names "gpu", "cpu", or "tpu"
    if device in ["cuda", "xla"]:
        device = "gpu"
    elif device != "cpu":
        device = "tpu"

    df = pd.DataFrame({"y": history, "ds": times})
    df["unique_id"] = 0

    tfm = timesfm.TimesFm(
        context_len=512,
        horizon_len=prediction_length,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend=device,
    )
    # train_state_unpadded_shape_dtype_struct
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    # TODO: pass frequency
    df_future = tfm.forecast_on_df(inputs=df, freq="H", value_name="y", num_jobs=-1)

    return ai4ts.model.Forecast(df_future.timesfm, "TimesFM", tfm)

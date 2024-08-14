import os.path

import pandas as pd


def read_df(input_file, parser=None):
    _, ext = os.path.splitext(input_file)
    if ext == ".parquet":
        df = pd.read_parquet(input_file)
    elif ext == ".csv":
        df = pd.read_csv(input_file)
    elif ext == ".hdf5":
        df = pd.read_hdf(input_file)
    elif ext == ".pickle":
        df = pd.read_pickle(input_file)
    else:
        msg = "Unknown input format for pandas: '%s'" % ext
        if parsier is not None:
            parser.error(msg)
        else:
            raise ValueError(msg)
    return df


def write_df(df, output_file, parser=None):
    _, ext = os.path.splitext(output_file)
    if ext == ".csv":
        df.to_csv(output_file)
    elif ext == ".parquet":
        df.to_parquet(output_file)
    elif ext == ".hdf5":
        df.to_hdf(output_file)
    elif ext == ".pickle":
        df.to_pickle(output_file)
    else:
        msg = "Unknown output format for pandas: '%s'" % ext
        if parsier is not None:
            parser.error(msg)
        else:
            raise ValueError(msg)

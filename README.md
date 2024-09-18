# Setup

Create virtual environment, using python 3.10 or greater:
```
cd path/to/time-series-ai-for-science
python -m venv env
source env/bin/activate
```

Install ai4ts dependencies and module to environment:
```
pip install -e ".[all]"
```

This will install all ml dependencies which can take over 8GB in space and a long time
to download. If doing minimal testing on a laptop without a GPU, you can run the following
instead:
```
pip install -e ".[lint]"
```

This will install statsforecast and the linter (useful for development) and none of
the ml libraries.

Using the editable ("-e") option of pip is useful for development as it does not create
a copy, so any changes are reflected immediately without the need to reinstall.

# Run

Generate an ARMA 2-2 sequence as a pandas dataframe with hourly data points
for 100 days (total 100 * 24 = 2400 data points):
```
ai4ts-arma-gen -n 2400 --phi 0.75 -0.25 --theta 0.65 0.35 -f h -o arma22.parquet
```

Compare arma forecast to several foundation model forecasts:
```
ai4ts-compare -l 24 -f H -i test/arima.yaml
```
New ARMA parameters can be added in `test/arima.yaml`. The `-l` option specifies the number
of data points to forecast, and the `-f` specifies the frequency to overlay on the data,
"H" is for hourly.

If running on a machine without GPU using the minimal install, you can compare only
the traditional non-ml models:
```
ai4ts-compare -l 24 -f H -i test/arima.yaml --models arma auto-arma
```

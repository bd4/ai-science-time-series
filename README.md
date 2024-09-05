# Setup

Create virtual environment:
```
cd path/to/time-series-ai-for-science
python -m venv env
source env/bin/activate
```

Install ai4ts dependencies and module to environment:
```
pip install -e .
```
Using the editable option of pip is does not create a copy so changes will be
reflected immediately without the need to re-install.

Note that this could take a long time to run, as there are a lot of dependencies.

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

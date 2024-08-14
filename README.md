# Setup

Create virtual environment:
```
cd path/to/time-series-ai-for-science
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Install ai4ts utilities and module to environment:
```
python setup.py develop
```
Using the develop option of setuptools is convenient because it will not create
a copy, so only needs to be done once and will see all changes.

# Run

Generate an ARMA 2-2 sequence as a pandas dataframe with hourly data points
for 100 days (total 100 * 24 = 2400 data points):
```
ai4ts-arma-gen -n 2400 --phi 0.75 -0.25 --theta 0.65 0.35 -f h -o arma22.parquet
```

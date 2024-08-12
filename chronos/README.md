# Chronos

## Setup

Create (if not already created) and activate venv:
```
cd path/to/time-series-ai-for-science
python -m venv env
source env/bin/activate
```

Install chronos:
```
pip install -r requirements.txt
```

## Test with synthetic ARMA data

Note: model will be downloaded as needed.
```
../bin/arma-gen.py -n 2400 --phi 0.75 -0.25 --theta 0.65 0.35 -o arma22.txt
./predict-sequence.py arma22.txt
```

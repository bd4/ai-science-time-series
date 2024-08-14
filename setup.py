from setuptools import setup

setup(
    name="ai4ts",
    version="0.1",
    description="Utilities for analyzing AI time series models",
    author="Bryce Allen",
    author_email="ballen@anl.gov",
    license="Apache 2.0",
    url="https://github.com/bd4/ai-science-time-series",
    packages=["ai4ts"],
    keywords=["ai", "time series"],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "statsmodels",
        "matplotlib",
        "fastparquet"
    ],
    entry_points={
        "console_scripts": [
            "ai4ts-arma-gen = ai4ts.arma:main",
        ],
    },
)

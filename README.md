# ML project template

This repository is a template directory for ML project and inspired by [upura](https://github.com/upura/ml-competition-template-titanic).

## Structures

```
.
├── configs
│   └── config.yml
├── data
│   ├── input
│   │   ├── sample_submission.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── output
├── docker
│   └── Dockerfile
├── features
├── logs
│   └── logger.py
├── models
├── notebooks
├── src
├── utils
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_base.py
│   ├── feature_create.py
│   └── convert_to_feather.py
├── .gitignore
├── LICENSE
├── README.md
└── run.py
```

## Directory Explanation

  - `config`: contains files that control feature settings, model parameters.
  - `data`
    - `input`: contains original data or feather files.
    - `output`: contains csv file for submission.
  - `docker`: contains `Dockerfile` and `requirements.txt`
  - `features`: contains features created by train and test data.
  - `fig`: contains some figures.
  - `logs`: contains logging data including features, a model, parameter and cv scores.
  - `models`: contains saved model.
  - `notebooks`: contains EDA codes.
  - `src`: contains model source codes and project-specific useful codes.
  - `utils`: contains generally useful codes.

## Setup

To prepare docker container, it is useful to use `Makefile` in the project directory.

- Build docker image: `make build`
- Run docker container: `make run`
- train model and predict: `python run.py`

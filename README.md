# ML project template

This repository is a template directory for ML project and inspired by [upura](https://github.com/upura/ml-competition-template-titanic).

## Structures

```
.
├── configs
│   └── config.json(not included here)
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
├── models
├── notebooks
├── src
├── utils
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

  - `config`:
    - `model`: model parameters
    - `config.yaml`: ML settings
  - `data`
    - `input`: contains original data or feather files.
    - `output`: contains csv file for submission.
  - `docker`: contains `Dockerfile` and `docker-compose.yml`
  - `features`: contains features created by train and test data.
    - `importance`: feature importances
  - `fig`: contains some figures.
  - `logs`: contains logging data including features, a model, parameter and cv scores.
  - `models`: contains saved model.
  - `notebooks`: contains EDA codes.
  - `src`: contains model source codes and project-specific useful codes.
  - `utils`: contains generally useful codes.
  - `requirements.txt`

## setup

- `docker-compose up -d`: prepare docker container
- `docker-compose run python bash`: start bash
## Actions

- `cd utils && python convert_to_feather.py`: Convert csv files to feather files.
- `python feature_create.py`: Create features in feather files.

- `cd .. && cd src && python run.py`: Start learning.

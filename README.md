# Suicide Rate Analysis & Machine Learning

**An end-to-end data analysis and regression project exploring historical suicide-rate data across countries, years, genders, age groups, and generations.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

## Overview

This project explores a historical dataset of suicide rates and related socioeconomic and demographic variables. It combines exploratory data analysis (EDA), data preprocessing, regression modeling, model comparison, and a basic serialized-model prediction workflow.

The goal is to practice a complete tabular machine-learning workflow and examine how different regression algorithms fit the available data. The results are exploratory and **must not be interpreted as individual suicide-risk predictions or used for clinical, policy, or other high-stakes decisions**.

## Project Highlights

- Explored **27,820 records** in the original dataset.
- Investigated patterns by country, year, gender, age group, and generation.
- Visualized distributions, group comparisons, temporal patterns, and relationships among numerical variables.
- Cleaned the dataset, removed selected columns, and filtered outliers using the IQR method on `suicide_rate`.
- Encoded categorical variables with `LabelEncoder` and scaled selected numerical variables with `RobustScaler`.
- Compared four regression algorithms using training/test \(R^2\) and RMSE.
- Serialized a model and preprocessing objects using Pickle.

## Dataset

The notebook loads a local CSV file named `master.csv`.

The dataset contains variables describing country, year, gender, age group, suicide count, population, suicide rate per 100,000 people, GDP, GDP per capita, and generation.

The schema resembles the public **Suicide Rates Overview 1985 to 2016** dataset. Confirm the exact source and include its original attribution before publishing the repository.

> The dataset contains aggregate historical statistics, not individual-level clinical information. Country-level and demographic-level associations do not establish causation.

## Workflow

### 1. Exploratory Data Analysis

The notebook uses Pandas, Matplotlib, and Seaborn to:

- Inspect dataset dimensions, column types, descriptive statistics, and unique values.
- Examine distributions of numerical variables.
- Compare recorded suicide counts across genders, age groups, and generations.
- Explore country-level suicide rates and year-level trends.
- Use scatter-matrix visualizations to inspect relationships among numerical features.

### 2. Data Cleaning and Preprocessing

The notebook performs the following operations:

1. Renames columns to make them easier to work with.
2. Drops the `HDI for year` and `country-year` columns.
3. Removes rows containing missing values after those columns are dropped.
4. Filters observations outside the IQR-based bounds of `suicide_rate`.
5. Encodes categorical columns using `LabelEncoder`.
6. Converts GDP values stored as strings into numeric values.
7. Scales `suicide_count`, `population`, `gdp_for_year`, and `gdp_per_capita` using `RobustScaler`.

The resulting dataset contains **25,774 rows** in the notebook's saved execution output.

### 3. Regression Modeling

The notebook uses an 80/20 train-test split with `random_state=42` and compares these models:

| Model | Configuration in notebook |
|---|---|
| Linear Regression | Default `LinearRegression` |
| Decision Tree Regressor | `max_depth=9` |
| Random Forest Regressor | `max_depth=9` |
| Bagging Regressor | `n_estimators=10` |

The notebook also explores Decision Tree and Random Forest depth settings and Bagging estimator counts through training/test score plots.

### 4. Evaluation

The notebook records:

- **\(R^2\)**: the coefficient of determination, indicating how much target variation is explained by the model relative to a baseline.
- **RMSE**: root mean squared error, measuring the typical scale of prediction errors in the target's units.

The notebook labels \(R^2\) as “Accuracy” in its result table. For a regression task, \(R^2\) is the more accurate term; it is not classification accuracy.

## Results From the Saved Notebook

The table below reproduces the metrics stored in the notebook output. These are results from that run, not independently reproduced or validated results.

| Model | Train \(R^2\) | Test \(R^2\) | Train RMSE | Test RMSE |
|---|---:|---:|---:|---:|
| Bagging Regressor | 0.999 | 0.997 | 0.243 | 0.485 |
| Random Forest Regressor | 0.991 | 0.986 | 0.928 | 1.094 |
| Decision Tree Regressor | 0.976 | 0.969 | 1.498 | 1.654 |
| Linear Regression | 0.284 | 0.274 | 8.179 | 8.006 |

The Bagging Regressor has the strongest recorded scores in this experiment. However, these high scores require careful interpretation.

### Important Evaluation Caveat: Potential Target Leakage

The target is `suicide_rate`, while the input features include `suicide_count` and `population`. Suicide rate per 100,000 people is mathematically derived from suicide count and population:

`suicide_rate ≈ suicide_count / population × 100,000`

This means the model receives information that directly determines the target. The high test \(R^2\) may therefore reflect target leakage rather than meaningful forecasting or generalization.

**Before claiming predictive performance, redesign the experiment** around a clearly defined use case. For example, remove features that directly construct the target, define whether the task is historical estimation or future forecasting, and use an appropriate time-based or grouped evaluation strategy. Recompute all metrics after this change.

## Visualizations

The notebook includes:

- Numerical feature histograms.
- Gender, age-group, and generation comparisons.
- Country-level rate plots.
- Year-level trend visualization.
- Scatter matrices for numerical variables.
- Actual-versus-predicted scatter plots.
- Training/test score curves for selected hyperparameters.
- Decision Tree feature-importance visualization.

For a polished GitHub presentation, export the most informative plots into a `results/` directory and embed them here using relative image paths.

## Tech Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn
- **Notebook environment:** Jupyter Notebook / Google Colab
- **Model persistence:** Pickle

## Run the Notebook

### Requirements

Python 3.x and the libraries used in the notebook.

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Steps

1. Clone or download this repository.
2. Place the source CSV file at the path expected by the notebook (`master.csv`), or update the path in the first notebook cell.
3. Install the dependencies.
4. Open `FinalProject.ipynb` in Jupyter Notebook or Google Colab.
5. Run the cells in order to perform EDA, preprocessing, training, and evaluation.

**Reproducibility note:** Some estimators in the notebook do not specify `random_state`, and the notebook saves preprocessing artifacts separately. Re-running the notebook may produce slightly different results. Set random seeds and package versions for a reproducible experiment.

## Suggested Repository Structure

```text
.
├── README.md
├── FinalProject.ipynb
├── master.csv                 # Keep out of Git if licensing or size requires
├── model.pkl                  # Generated model artifact, if intentionally included
├── robust.pkl                 # Generated scaler, if intentionally included
└── label.pkl                  # Generated label encoders, if intentionally included
```

This structure reflects the files and artifacts referenced by the notebook. Only commit files that actually exist and that you are permitted to redistribute. Do not commit sensitive data or unnecessary large files.

## Limitations and Next Steps

- Rework the target/features to address potential target leakage.
- Use a time-based split if the intended task is forecasting future rates.
- Fit encoders and scalers on the training set only, then apply them to validation/test data.
- Put preprocessing and the estimator into a single scikit-learn `Pipeline`.
- Add cross-validation and compare against a simple baseline.
- Report error analysis and clearly document the evaluation protocol.
- Refactor the notebook's interactive prediction code into a tested function or small application.
- Add a pinned `requirements.txt`, dataset attribution, and exported figures.

## Responsible Use

This project is for educational analysis of historical aggregate data. It does not predict an individual's suicide risk, explain why an individual may be at risk, or establish causal relationships. Model outputs should not be used to make decisions about individuals or communities.

If you or someone else may be in immediate danger, contact local emergency services or a qualified crisis-support professional.

## Author

**Thắng Nguyễn Hữu**

- GitHub: [@Huuthanga](https://github.com/Huuthanga)

---

*This project demonstrates an exploratory workflow for data analysis and regression modeling, including the importance of checking for leakage before interpreting model performance.*

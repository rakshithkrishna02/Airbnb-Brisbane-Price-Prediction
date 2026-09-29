# Airbnb Brisbane Price Prediction

Predicting the nightly price of Airbnb listings in Brisbane, Australia, using machine learning. Built for a university Kaggle competition (BUSA8001, Macquarie University, 2026), where our team **improved from rank 47 to rank 39** by reducing prediction error from **27.4% to 22.2% MAPE**.

## The problem

Hosts often guess their nightly price, which means leaving money on the table or scaring guests away. This project builds a regression model that estimates a fair nightly price from a listing's features, such as property size, location, availability, reviews and host behaviour. The same approach could support pricing tools for hosts, property investors, and short-term rental platforms.

- **Data:** 3,735 training listings and 1,601 test listings, 64 raw columns
- **Target:** nightly `price` (right-skewed, ranging from $36 to $5,000, median $193)
- **Metric:** Mean Absolute Percentage Error (MAPE), the official Kaggle metric

## Approach

**1. Exploratory data analysis.** Profiled missing values, distributions and relationships with price. Capacity features were the strongest numeric predictors (correlation with price: `accommodates` 0.52, `bedrooms` 0.50), and price varied clearly by room type and neighbourhood. Selected 29 features (23 numeric, 1 ordinal, 5 nominal).

**2. Data cleaning and feature engineering** *(my section)*
- Converted text-based numeric fields (`$`, `%`, commas) into proper numeric types
- Engineered five new features: `guests_per_bedroom`, `bathrooms_per_guest`, `recent_review_share`, `availability_gap_short_long` and `review_score_mean`
- Imputed missing values with the median (numeric) and mode (categorical), using training-set values only to avoid data leakage
- Encoded `host_response_time` as ordinal and one-hot encoded nominal features, keeping the top 5 categories and grouping the rest as "Other"
- Clipped outliers at the 1st and 99th percentiles and standardised features with `StandardScaler`
- Final model-ready dataset: 49 features

**3. Modelling and tuning.** Compared three models using 5-fold `GridSearchCV` optimising MAPE, plus a 20% held-out validation set.

| Model | Training MAPE | Validation MAPE |
| --- | --- | --- |
| Ridge Regression | 36.88% | 39.12% |
| Decision Tree | 28.83% | 32.51% |
| **Random Forest** | 14.24% | **28.71%** |

**4. Improvement.** Because price was heavily right-skewed, we trained the Random Forest on `log(1 + price)` and converted predictions back to dollars. This stopped a few luxury listings from dominating the model.

## Results

| Submission | Validation MAPE | Kaggle score (MAPE) | Leaderboard rank |
| --- | --- | --- | --- |
| Random Forest, raw price | 28.71% | 0.274 | 47 |
| **Random Forest, log-transformed price** | **22.77%** | **0.222** | **39** |

The log transformation cut validation error by about 5.9 percentage points.

## Tech stack

Python, Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter Notebook

## How to run

1. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download `train.csv` and `test.csv` from the competition and place them in a `Dataset/` folder (the data is not included in this repository).
3. Open `Airbnb Price Prediction.ipynb` in Jupyter and run all cells.

## Team

Group project by team **BUSA8001_Gaggle**:

- Harisha Sundaram: Problem description and EDA
- **Rakshith Krishna Suresh Hema: Data cleaning and feature engineering**
- Jarernrat Srimaothongsuk: Model fitting, tuning and prediction

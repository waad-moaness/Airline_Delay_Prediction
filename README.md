# Airline Delay Prediction

## 1. Problem Statement

Flight delays result in massive logistical challenges and financial costs for airlines and airports. This project focuses on predicting the **magnitude of arrival delays** (in minutes) for specific airline-airport combinations.

Unlike simple classification (delayed vs. on-time), this project treats the problem as a **Regression task**. By predicting the total delay minutes based on flight volume, carrier, and historical patterns, stakeholders can better estimate operational strain and allocate resources efficiently.

## 2. Dataset Description

The project utilizes the **Airline Delay Cause** dataset, which consists of aggregated monthly flight statistics.

* **Source:** `https://www.kaggle.com/datasets/mahmoudhassanmahmoud/airline-delay-20032022`
* **Type:** Aggregated Data (Monthly summaries per Carrier per Airport).
* **Target Variable:** `arr_delay` (Total arrival delay in minutes).
* **Input Features:**
    * **Numerical:** `year`, `month`, `arr_flights` (Total flights arriving), `arr_cancelled`, `arr_diverted`.
    * **Categorical:** `carrier` (Airline Code), `carrier_name`, `airport` (Airport Code), `airport_name`.

## 3. EDA Summary

Exploratory Data Analysis (EDA) revealed specific characteristics of the data that influenced the modeling strategy:

* **Target Distribution:** The `arr_delay` variable was extremely right-skewed (a long tail of massive delays). A **Log Transformation** was crucial to stabilize variance and improve model performance.
* **Feature Correlations:**
    * `arr_flights` (Flight Volume) showed the highest correlation with total delay (~0.91 importance in Decision Trees), indicating that volume is the primary driver of cumulative delay.
* **Seasonality:** `month` and `year` were identified as relevant predictors, capturing seasonal travel trends.

## 4. Modeling Approach & Metrics

### Modeling Approach
The project followed a robust Machine Learning pipeline using `scikit-learn` and `xgboost`:

1.  **Preprocessing:**
    * **Categorical Features:** Handled using `OneHotEncoder` (ignoring unknown categories).
    * **Numerical Features:** Scaled using `StandardScaler`.
    * **Target Variable:** Log-transformed (`log1p`) before training and inverse-transformed (`expm1`) for interpretation.
2.  **Model Selection:** Three models were trained and evaluated:
    * **Linear Regression:** Baseline model.
    * **Decision Tree Regressor:** Hyperparameter tuned (`max_depth`, `min_samples_leaf`).
    * **XGBoost Regressor:** The final model chosen for its superior performance and ability to capture non-linear relationships.

### Metrics
The models were evaluated using **Root Mean Squared Error (RMSE)** and **R-squared (R2)** on the log-transformed target.

| Model | RMSE (Log Scale) | R2 Score |
| :--- | :--- | :--- |
| Linear Regression | ~1.51 | 0.38 |
| Decision Tree (Tuned) | ~0.84 | 0.80 |
| **XGBoost (Final)** | **~0.78** | **0.83** |

**Final Model:** The XGBoost model achieved an **R2 score of ~0.83**, explaining 83% of the variance in the delay data.

## 5. How to Run Locally and Via Docker

### Prerequisites
* Python 3.9+
* Docker (optional)
* Pipenv (recommended for dependency management)

### Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/waad-moaness/Airline_Delay_Prediction.git
    cd Airline_Delay_Prediction
    ```
2.  **Prerequisites**
    You must have Python and Pipenv installed.
    ```bash
    pip install pipenv
    ```

3.  **Install Dependencies:**
    Use `sync` to install the exact versions from the lock file.
    ```bash
    pipenv sync
    ```

4.  **Activate the Virtual Environment:**
    ```bash
    pipenv shell
    ```

5.  **Run the Prediction Service:**
    ```bash
    python predict.py
    ```
    *The service will start on `http://0.0.0.0:9696`*

### Running Via Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t airline-delay-regression .
    ```

2.  **Run the container:**
    ```bash
    docker run -p 9696:9696 airline-delay-regression
    ```

## 6. API Usage Example

The model expects a JSON payload containing the aggregated monthly stats for a specific carrier and airport.

**Endpoint:** `POST /predict`

**Example Request Body:**
```json
{
    "year": 2022,
    "month": 5,
    "arr_flights": 181.0,
    "arr_cancelled": 0.0,
    "arr_diverted": 0.0,
    "carrier": "9e",
    "carrier_name": "endeavor_air_inc",
    "airport": "ags",
    "airport_name": "augusta_ga_augusta_regional_at_bush_field"
}
```

## 7. Known Limitations / Next Steps

### Limitations

* **Aggregated Data:** The model relies on monthly summaries. It cannot predict delays for a specific single flight (e.g., "Flight 101 on Tuesday") but rather the general performance of an airline at an airport for the month.
* **Log Scale Interpretation:** Because the model is trained on log-transformed targets to handle skewness, errors in prediction can be magnified when converted back to real minutes for very large delays.

### Next Steps

* **Granular Data:** Incorporate daily or individual flight-level data to allow for real-time, single-flight predictions.
* **Weather Integration:** Add specific weather data (precipitation, wind speed) for the airport/month to improve regression accuracy.
* **Cloud Deployment:** Deploy the containerized service to a cloud provider like AWS Elastic Beanstalk or Google Cloud Run.


## Demo Video

https://github.com/user-attachments/assets/58430514-cf3a-490b-b72c-70dd16473a46




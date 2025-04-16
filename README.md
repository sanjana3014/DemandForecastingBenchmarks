# DemandForecastingBenchmarks

# DemandForecastingBenchmarks

## Overview

**DemandForecastingBenchmarks** is a machine learning project that predicts **weekly sales** using deep learning and ensemble techniques. Built on a real-world e-commerce dataset spanning **2022 to 2024**, this project applies rigorous preprocessing and a comparative study of advanced models including:

- Recurrent Neural Networks (RNNs): Bi-LSTM, GRU, Bi-GRU  
- XGBoost Regressor  
- MLP Regressor

We implement extensive **hyperparameter tuning** and **result visualization** to evaluate forecasting performance.

---

## Problem Statement

The objective is to **forecast future weekly sales** of various sellers and brands using past sales trends. This helps the e-commerce platform make informed decisions for:

- Inventory planning  
- Marketing strategy  
- Pricing optimization  
- Demand prediction

---

## Dataset

- **Source**: Proprietary Wildberries (Russian E-commerce platform) UGG sales dataset
- **Format**: 36 monthly `.csv` files (2022–2024)
- **Preprocessing**:
  - Column translation (Russian → English)
  -  Removal of 60+ irrelevant attributes
  - Feature engineering (lag, sales trend, weekly grouping)
  - Aggregation by `Seller` and `Brand`
  - Time-series reshaping
  - Scaling using `MinMaxScaler`
  - Train-Test-Validation Split: 80:10:10

---

## Models Used

| Model        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| Bi-LSTM      | Bidirectional LSTM capturing both past and future context in sequences      |
| GRU          | Gated Recurrent Unit model for faster training and fewer parameters         |
| Bi-GRU       | Bidirectional version of GRU for improved context learning                  |
| XGBoost      | Tree-based ensemble with gradient boosting for tabular regression           |
| MLPRegressor | Classic feedforward neural network used for regression on flat features     |

---

## Hyperparameters Evaluated

- **RNNs (Bi-LSTM, GRU, Bi-GRU)**:
  - `units`: 32, 64
  - `dropout`: 0.2, 0.3
  - `lr`: 0.001, 0.0005
  - `batch_size`: 16, 32
  - `epochs`: 50, 100

- **XGBoost**:
  - `n_estimators`: 50, 100
  - `max_depth`: 3, 5
  - `learning_rate`: 0.1, 0.05
  - `subsample`: 0.8
  - `reg_alpha`: 0, 0.1
  - `reg_lambda`: 1
  - `colsample_bytree`: 1

- **MLPRegressor**:
  - `hidden_layer_sizes`: (64, 32), (128, 64)
  - `activation`: relu, tanh
  - `solver`: adam
  - `alpha`: 0.0001, 0.001
  - `learning_rate_init`: 0.001

---

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

---

## Model Comparison Results


| Model    | Test RMSE | Test MAE |
|----------|-----------|----------|
| **Bi-GRU**   | **3390.23**  | **1162.85** |
| GRU      | 2639.54   | 773.73   |
| Bi-LSTM  | 3837.59   | 1134.89  |
| MLP      | 3151.38   | 992.47   |
| XGBoost  | 7743.18   | 2936.74  |

**Bi-GRU** achieved the best generalization ability and performance on validation and test sets.

---

## Visualizations

- Daily & Weekly sales line plots  
- Model prediction vs actual sales curves  
- Comparative bar chart of RMSE across models

## Final Model Details

**Best Performing Model**: Bi-GRU  
**Best Hyperparameters**:
- `units`: 64  
- `dropout`: 0.3  
- `learning_rate`: 0.0005  
- `batch_size`: 16  
- `epochs`: 100  

**Test Performance**:
- **RMSE**: 3390.23  
- **MAE**: 1162.85

---

## Acknowledgements

Special thanks to Zhanna Latypova, my teammate, for the preprocessing efforts and project collaboration.

---

## License

MIT License — *Free to use, share and modify with attribution.*



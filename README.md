#  Energy Demand Forecasting — EV Charging Stations

Forecasting hourly electric vehicle (EV) charging demand using deep learning and graph-based spatio-temporal models, benchmarked against published research results.

## Overview

This project predicts hourly EV charging energy demand (kWh) across multiple charging station locations in Boulder, Colorado, using historical charging session data. The goal was to explore how much predictive power comes from adding **spatial relationships between locations** on top of standard time-series modeling.

Four progressively more advanced models were built and compared:

| Model | Description |
|---|---|
| **LSTM** | Baseline sequential model using time-based features only |
| **BiLSTM** | Bidirectional LSTM to capture patterns from both past and future context in training |
| **GCN-LSTM** | Graph Convolutional Network + LSTM — incorporates spatial relationships between charging station locations (ZIP codes) |
| **GAT-LSTM (Proposed)** | Graph Attention Network + LSTM — learns *which* neighboring locations matter most, rather than treating all spatial neighbors equally |

##  Dataset

- **Source:** Boulder, Colorado EV Charging Station session data (148,136 records, 17 columns)
- **Key fields used:** session start/end time, energy delivered (kWh), station location (ZIP code)
- **Processing:** Raw session-level data aggregated into hourly time series per location, with missing hours filled to maintain a continuous timeline

##  Feature Engineering

- Cyclical time encodings (hour, day-of-week, month) using sine/cosine transforms to help the model understand time as continuous rather than categorical
- Lag features to capture daily and weekly demand patterns
- Geographic coordinates per ZIP code used to build a spatial adjacency graph (via Euclidean distance) for the graph-based models
- 7-day (168-hour) lookback window for all sequence models

## Model Architecture (Proposed: GAT-LSTM)

- Custom Graph Attention Layer with multi-head attention (4 heads) to weigh the influence of neighboring charging locations dynamically
- Combined with stacked LSTM layers to capture temporal dependencies
- Trained with Huber loss (robust to demand spikes) and Adam optimizer with learning rate scheduling
- Early stopping on validation loss to prevent overfitting

##  Results

| Model | RMSE (kWh) | MAE (kWh) | RMSE (normalized) | MAE (normalized) |
|---|---|---|---|---|
| LSTM | 25.96 | 18.05 | 0.0628 | 0.0398 |
| BiLSTM | 25.32 | 17.04 | 0.0612 | 0.0375 |
| GCN-LSTM | 12.10 | 6.36 | 0.0571 | 0.0140 |
| **GAT-LSTM (Proposed)** | **11.82** | **6.41** | **0.0558** | **0.0141** |

*Note: LSTM/BiLSTM metrics reflect aggregate city-level demand; GCN-LSTM/GAT-LSTM metrics reflect per-location demand (ZIP 80301). Normalized RMSE/MAE are directly comparable across all four models.*

**Benchmark comparison:** the proposed GAT-LSTM model achieved a normalized RMSE of 0.0558, a **3.3% improvement** over the published benchmark from Alaraj et al. (2025), which reported 0.0577.

Adding spatial graph structure (GCN-LSTM, GAT-LSTM) substantially outperformed sequence-only models (LSTM, BiLSTM) — roughly halving both RMSE and MAE — confirming that EV charging demand at one location is meaningfully influenced by demand patterns at nearby locations. The attention-based GAT-LSTM further improved on GCN-LSTM by 2.3% (RMSE), since it learns *which* neighboring locations matter most rather than weighting all neighbors equally.

## Tech Stack

`Python` · `TensorFlow / Keras` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `SciPy`

##  Repository Structure

```
├── energy_demand_forecasting.ipynb   # Full analysis, modeling, and evaluation notebook
├── README.md
└── requirements.txt
```

##  How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `energy_demand_forecasting.ipynb` in Jupyter or Google Colab
4. Dataset: Boulder EV Charging Station data (see notebook for source link)

##  Future Improvements

- Extend the spatial graph beyond ZIP-code-level centroids to individual station-level coordinates
- Incorporate weather and local event data as external regressors
- Deploy as a real-time forecasting dashboard

---
*Final semester project — MSc Data Science & Artificial Intelligence*

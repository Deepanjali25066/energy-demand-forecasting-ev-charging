# Energy Demand Forecasting — EV Charging Stations

Forecasting hourly electric vehicle (EV) charging demand using deep learning and graph-based spatio-temporal models, benchmarked against published research results.

##  Overview

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

##  Model Architecture (Proposed: GAT-LSTM)

- Custom Graph Attention Layer with multi-head attention (4 heads) to weigh the influence of neighboring charging locations dynamically
- Combined with stacked LSTM layers to capture temporal dependencies
- Trained with Huber loss (robust to demand spikes) and Adam optimizer with learning rate scheduling
- Early stopping on validation loss to prevent overfitting

## 📈 Results

ModelRMSE (kWh)MAE (kWh)RMSE (normalized)MAE (normalized)LSTM25.9618.050.06280.0398BiLSTM25.3217.040.06120.0375GCN-LSTM12.106.360.05710.0140GAT-LSTM (Proposed)11.826.410.05580.0141

Adding spatial graph structure (GCN-LSTM, GAT-LSTM) substantially outperformed sequence-only models (LSTM, BiLSTM), confirming that EV charging demand at one location is meaningfully influenced by demand patterns at nearby locations.

Results were also benchmarked against Alaraj et al. (2025) for validation against published literature.

##  Tech Stack

`Python` · `TensorFlow / Keras` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `SciPy`

## 📁 Repository Structure

```
├── energy_demand_forecasting.ipynb   # Full analysis, modeling, and evaluation notebook
├── README.md
└── requirements.txt
```

## 🚀 How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `energy_demand_forecasting.ipynb` in Jupyter or Google Colab
4. Dataset: Boulder EV Charging Station data (see notebook for source link)

## 🔭 Future Improvements

- Extend the spatial graph beyond ZIP-code-level centroids to individual station-level coordinates
- Incorporate weather and local event data as external regressors
- Deploy as a real-time forecasting dashboard

---
*Final semester project — MSc Data Science & Artificial Intelligence*

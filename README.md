# 🧭 Crypto Compass

A real-time crypto direction predictor and market analysis dashboard built with Streamlit.

## 👋 Overview

Crypto Compass turns order-book data into a short-term directional signal. Its Transformer model reads a rolling window of Order Book Imbalance (OBI) values and predicts market direction ahead. The app is designed to help you explore live market pressure and compare it with recent prediction behavior.

Use it to:

- 📡 Follow live predictions as new data arrives
- 📂 Explore crypto data from a CSV upload
- 📊 Compare market features and model predictions
- 🔎 Review model performance and prediction confidence

## ✨ Explore the app

### 🏠 Front page

Start here to upload a CSV file or switch on the **Live Prediction** option.

### ⚡ Live Prediction

The live view uses Binance BTC order-book data and refreshes every **250 ms**.

**Left panel — rolling input and signal**

- Reads 12 sequential OBI values, representing approximately 3 seconds of history.
- On each refresh, adds the latest value and drops the oldest one.
- Predicts market direction 1 second ahead.

The direction target is based on the log-ratio of future and current mid-price:

```text
signal = log(mid_price_after_1s / latest_mid_price)
```

At a 250 ms interval, a 1-second horizon corresponds to four future rows.

**Right panel — recent predictions**

Keep an eye on up to five recent results, including their timestamp, predicted direction, and probability.

### 📈 Overview

Upload a CSV to open the analysis dashboard:

**Analysis tab**

- Inspect original order-book fields, including bid/ask prices and volumes.
- Compare OBI, spread, and mid-price in charts.
- Explore why OBI is used as the model input.

**Prediction analysis tab**

- Compare model predictions with observed market movement.
- Review prediction accuracy and model behavior on the uploaded data.
- Examine the resources used to generate predictions.

## 🧠 Transformer model

The model converts a short sequence of OBI observations into one of two directional classes. Follow the path from input to prediction:

```text
12 recent OBI values
        ↓
Truncated-normal positional embeddings
        ↓
2 Transformer encoder layers
        ↓
2-class output: DOWN (0) or UP (1)
```

### 🔍 What happens at each step?

1. **Read the sequence:** The model receives one feature—OBI—across 12 time steps.
2. **Add position information:** Truncated-normal positional embeddings help represent where each value occurs in the sequence.
3. **Attend to the sequence:** Self-attention helps the model weigh relationships between values; the encoder has two layers and one attention head.
4. **Predict a class:** The output contains two classes, labeled `0` (**DOWN**) and `1` (**UP**), along with predicted probabilities. The target uses the four-row-ahead mid-price log-ratio: `log(mid_price_after_4_rows / latest_mid_price)`.

<details>
<summary>🛠️ View model specifications</summary>

| Parameter | Value |
|---|---:|
| Input features | 1 |
| Sequence length | 12 |
| Model dimension (`d_model`) | 32 |
| Attention heads (`n_head`) | 1 |
| Transformer layers (`n_layers`) | 2 |
| Feed-forward dimension | 64 |
| Output classes | 2 |
| Dropout | 0.1 |

</details>

## 📊 Model performance

Reported evaluation metrics:

| Metric | Score |
|---|---:|
| Accuracy | 0.6163 |
| Loss | 0.6553 |
| ROC AUC | 0.6365 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.63 | 0.78 | 0.70 | 213,271 |
| 1 | 0.58 | 0.39 | 0.47 | 159,805 |
| **Accuracy** | — | — | **0.62** | **373,076** |
| **Macro average** | 0.60 | 0.59 | 0.58 | 373,076 |
| **Weighted average** | 0.61 | 0.62 | 0.60 | 373,076 |

<details>
<summary>💡 Why use OBI?</summary>

OBI summarizes the relative pressure between bid and ask sides of the order book and is scaled between -1 and 1. In the project's feature experiments, it was selected as the sole model input because it captures short-term directional pressure efficiently and remains easy to interpret.

</details>

## 🚀 Quick start

1. Install the project dependencies.
2. From the directory containing `app.py`, start Streamlit:

   ```bash
   streamlit run app.py
   ```

3. Upload a CSV file or enable live prediction.
4. Explore the dashboard and prediction results.

## ⚠️ Notes

- Live predictions update frequently and are intended for monitoring short-term signals.
- Historical analysis is available after loading a CSV file.
- Model metrics describe the reported evaluation and do not guarantee future performance.

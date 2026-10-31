from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_v3 import Transformer

SEQ_LEN = 12
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
CHECKPOINT_PATH = MODEL_DIR / "transformer_checkpoint.pth"


def csv_to_df(file):
    """Load a 40- or 42-column order-book CSV into the training column layout.

    A 42-column file has two leading metadata columns, which are dropped.
    After that, positions 0 and 20 are best bid/ask prices, bid volumes are
    at odd positions 1..19, and ask volumes are at odd positions 21..39.
    """
    if file is None:
        raise ValueError("A CSV file path or file-like object is required.")

    df = pd.read_csv(file)
    if df.empty:
        raise ValueError("Input CSV contains no data rows.")

    if df.shape[1] == 42:
        df = df.iloc[:, 2:].copy()
    elif df.shape[1] == 40:
        df = df.copy()
    else:
        raise ValueError(
            f"Input CSV must contain 40 or 42 columns; found {df.shape[1]}."
        )

    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV order-book columns must contain numeric data.") from exc

    if not np.isfinite(df.to_numpy(dtype=np.float64)).all():
        raise ValueError("CSV contains missing or non-finite values.")

    bid_volume_columns = list(range(1, 20, 2))
    ask_volume_columns = list(range(21, 40, 2))
    volume_columns = bid_volume_columns + ask_volume_columns

    if (df.iloc[:, volume_columns] < 0).any().any():
        raise ValueError("Bid and ask volumes must be non-negative.")
    if (df.iloc[:, [0, 20]] <= 0).any().any():
        raise ValueError("Best bid and ask prices must be positive.")
    if (df.iloc[:, 0] >= df.iloc[:, 20]).any():
        raise ValueError("Best bid price must be lower than best ask price.")

    return df


def feature_engineering(file=None):
    """Calculate row-wise OBI using the same volume columns as features_v2.py."""
    df = csv_to_df(file)
    bid_volume_columns = list(range(1, 20, 2))
    ask_volume_columns = list(range(21, 40, 2))

    mean_bid_volume = df.iloc[:, bid_volume_columns].mean(axis=1)
    mean_ask_volume = df.iloc[:, ask_volume_columns].mean(axis=1)
    total_volume = mean_bid_volume + mean_ask_volume
    obi = (mean_bid_volume - mean_ask_volume).div(total_volume.where(total_volume != 0))#type:ignore
    return obi.fillna(0.0).rename("OBI")

def _validate_sequence_length(sequence_length):
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length <= 0
    ):
        raise ValueError("sequence_length must be a positive integer.")


def _load_transformer(sequence_length, device):
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Transformer checkpoint not found: {CHECKPOINT_PATH}")

    model = Transformer(
        n_features=1,
        seq_len=sequence_length,
        d_model=32,
        n_head=1,
        n_layers=2,
        dim_foreforward=64,
        dim_output=2,
        dropout=0.1,
    ).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict_from_tensor(inputs, model, device):
    with torch.inference_mode():
        logits = model(inputs.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        prediction_index = int(torch.argmax(logits, dim=1).item())

    prediction = "DOWN" if prediction_index == 0 else "UP"
    return [prediction], [probabilities]


class Rolling_prediction:
    """Make rolling predictions from one live OBI observation per call."""

    def __init__(self, sequence_length=SEQ_LEN):
        
        _validate_sequence_length(sequence_length)
        
        if not SCALER_PATH.is_file():
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

        self.sequence_length = sequence_length
        self.window = deque(maxlen=sequence_length)
        self.scaler = joblib.load(SCALER_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_transformer(sequence_length, self.device)

    def update_obi_window(self, new_obi):
        """Append an OBI sample; return the full window once it is ready."""
        value = float(new_obi)
        if not np.isfinite(value):
            raise ValueError("new_obi must be finite.")

        self.window.append(value)
        return list(self.window) if len(self.window) == self.sequence_length else None

    def tensor_formation(self):
        """Scale the current live window into model input shaped (1, seq_len, 1)."""
        if len(self.window) < self.sequence_length:
            raise ValueError(
                f"Need {self.sequence_length} OBI samples; received {len(self.window)}."
            )

        values = pd.DataFrame({"OBI": self.window})
        scaled = self.scaler.transform(values).astype(np.float32)
        return torch.from_numpy(scaled).unsqueeze(0)

    def predict(self, new_obi):
        """Consume one live OBI sample and predict when the window is full.

        Returns None during the initial warm-up, then returns
        (["DOWN" or "UP"], [[P(DOWN), P(UP)]]) on each call.
        """
        if self.update_obi_window(new_obi) is None:
            return None

        inputs = self.tensor_formation()
        return _predict_from_tensor(inputs, self.model, self.device)

    def transformer_model(self, new_obi):
        """Compatibility alias for predict(new_obi)."""
        return self.predict(new_obi)


def tensor_formation(file, sequence_length=SEQ_LEN):
    """Return the latest scaled OBI window from a CSV as a tensor."""
    _validate_sequence_length(sequence_length)
    
    obi = feature_engineering(file)
    
    if len(obi) < sequence_length:
        raise ValueError(
            f"At least {sequence_length} valid CSV rows are required; found {len(obi)}."
        )
    
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    values = pd.DataFrame({"OBI": obi.iloc[-sequence_length:].to_numpy()})
    scaled = scaler.transform(values).astype(np.float32)
    
    return torch.from_numpy(scaled).unsqueeze(0)


def transformer_model(file, sequence_length=SEQ_LEN):
    """Predict the latest CSV OBI window (DOWN/UP and both probabilities)."""
    _validate_sequence_length(sequence_length)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = tensor_formation(file, sequence_length=sequence_length)
    
    model = _load_transformer(sequence_length, device)
    
    return _predict_from_tensor(inputs, model, device)


def predict_obi_stream(obi_stream, sequence_length=SEQ_LEN, batch_size=4096):
    """Run rolling Transformer predictions over an already-extracted OBI stream.

    The first sequence_length - 1 rows have no prediction because a complete
    input window is not available yet. Every subsequent prediction uses the
    latest sequence_length OBI observations, matching Rolling_prediction.

    Returns a DataFrame aligned to the input index with:
        MODEL_PREDICTION, DOWN_PROBABILITY, UP_PROBABILITY
    """
    _validate_sequence_length(sequence_length)

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if isinstance(obi_stream, pd.DataFrame):
        if "OBI" not in obi_stream.columns:
            raise ValueError("obi_stream DataFrame must contain an 'OBI' column.")
        obi_values = obi_stream["OBI"]
    else:
        obi_values = pd.Series(obi_stream, name="OBI")

    obi_values = pd.to_numeric(obi_values, errors="coerce")

    if obi_values.isna().any() or not np.isfinite(obi_values.to_numpy(dtype=np.float64)).all():
        raise ValueError("OBI stream must contain only finite numeric values.")

    if len(obi_values) < sequence_length:
        raise ValueError(
            f"At least {sequence_length} OBI observations are required; "
            f"received {len(obi_values)}."
        )

    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(SCALER_PATH)
    model = _load_transformer(sequence_length, device)

    # Scale each OBI observation with the same fitted training scaler.
    scaled = scaler.transform(
        pd.DataFrame({"OBI": obi_values.to_numpy(dtype=np.float64)})
    ).astype(np.float32)

    # Build overlapping windows. The first prediction corresponds to input row
    # sequence_length - 1, exactly as in Rolling_prediction.
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled[:, 0], window_shape=sequence_length
    )

    predictions = np.full(len(obi_values), None, dtype=object)
    down_probabilities = np.full(len(obi_values), np.nan, dtype=np.float64)
    up_probabilities = np.full(len(obi_values), np.nan, dtype=np.float64)

    first_prediction_row = sequence_length - 1

    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            end = min(start + batch_size, len(windows))

            batch_windows = np.ascontiguousarray(
                windows[start:end, :, np.newaxis],
                dtype=np.float32,
            )
            inputs = torch.from_numpy(batch_windows).to(device)

            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

            output_start = first_prediction_row + start
            output_end = first_prediction_row + end

            predictions[output_start:output_end] = np.where(
                predicted_classes == 0, "DOWN", "UP"
            )
            down_probabilities[output_start:output_end] = probabilities[:, 0]
            up_probabilities[output_start:output_end] = probabilities[:, 1]

    return pd.DataFrame(
        {
            "MODEL_PREDICTION": predictions,
            "DOWN_PROBABILITY": down_probabilities,
            "UP_PROBABILITY": up_probabilities,
        },
        index=obi_values.index,
    )


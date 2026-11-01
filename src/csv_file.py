from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Default input file. Change this path if your dataset is stored elsewhere.
DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "binance_book_snapshot_25_2024-01-01_BTCUSDT.csv.gz"
)

PERCENTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
LEVELS_USED = 10
RESAMPLE_INTERVAL = "250ms"
PREDICTION_HORIZON_ROWS = 4  # 4 x 250 ms = 1 second


def summarize_csv(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract and summarize a Tardis Binance book_snapshot_25 CSV.

    Uses the first 10 bid and ask levels, preserving the existing
    Crypto Compass feature definitions.

    Processing:
      1. Sort snapshots by local_timestamp (microseconds).
      2. Calculate average bid/ask prices and quantities across 10 levels.
      3. Calculate OBI, spread and mid-price.
      4. Resample the feature stream to 250 ms using the last observation
         in each interval, then forward-fill empty intervals.
      5. Calculate the actual future 1-second return and direction.

    Returns:
        feature_percentiles:
            Percentile summary for OBI, SPREAD and MID_PRICE.
        orderbook_describe:
            Descriptive statistics for average ask/bid prices and volumes.
        feature_stream:
            Chronological 250 ms feature stream, including OBI and actual
            future market direction.

    Direction convention matches the existing binary target:
        1 = UP   (future return > 0)
        0 = DOWN (future return <= 0)
        <NA> = UNKNOWN (future price unavailable)
    """
    # Accept either a local filesystem path or a remote HTTP(S) URL.
    # Keep URLs as strings: pathlib.Path normalizes https:// into https:/.
    csv_source = str(csv_path).strip()

    if csv_source.startswith(("http://", "https://")):
        # Tardis downloads are commonly .csv.gz; pandas can decompress them.
        compression = "gzip" if csv_source.split("?", 1)[0].lower().endswith(".gz") else "infer"
        df = pd.read_csv(csv_source, compression=compression)
    else:
        local_path = Path(csv_source)
        if not local_path.exists():
            raise FileNotFoundError(f"CSV file not found: {local_path}")

        df = pd.read_csv(local_path)

    # Validate metadata and the first 10 levels of the order book.
    required_columns = ["timestamp", "local_timestamp"]

    for side in ("asks", "bids"):
        for level in range(LEVELS_USED):
            required_columns.extend(
                [
                    f"{side}[{level}].price",
                    f"{side}[{level}].amount",
                ]
            )

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        preview = ", ".join(missing[:8])
        more = " ..." if len(missing) > 8 else ""
        raise ValueError(
            "This CSV does not have the expected Tardis book_snapshot_25 "
            f"columns. Missing: {preview}{more}"
        )

    if df.empty:
        raise ValueError("The input CSV contains no rows.")

    # Tardis local_timestamp is in microseconds.
    # Stable sorting preserves original order when timestamps are equal.
    df = (
        df.sort_values("local_timestamp", kind="stable")
        .reset_index(drop=True)
    )

    # Ensure order-book columns are numeric.
    numeric_columns = [
        f"{side}[{level}].{field}"
        for side in ("asks", "bids")
        for level in range(LEVELS_USED)
        for field in ("price", "amount")
    ]
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    df["local_timestamp"] = pd.to_numeric(
        df["local_timestamp"], errors="coerce"
    )

    # Rows without a usable local capture timestamp cannot be resampled.
    df = df.dropna(subset=["local_timestamp"]).copy()
    if df.empty:
        raise ValueError("No valid local_timestamp values were found.")

    # Extract the first 10 levels on each side.
    bid_price_columns = [f"bids[{i}].price" for i in range(LEVELS_USED)]
    ask_price_columns = [f"asks[{i}].price" for i in range(LEVELS_USED)]
    bid_volume_columns = [f"bids[{i}].amount" for i in range(LEVELS_USED)]
    ask_volume_columns = [f"asks[{i}].amount" for i in range(LEVELS_USED)]

    # Preserve the existing averaging definitions.
    bid_price = df[bid_price_columns].mean(axis=1)
    ask_price = df[ask_price_columns].mean(axis=1)
    bid_vol = df[bid_volume_columns].mean(axis=1)
    ask_vol = df[ask_volume_columns].mean(axis=1)

    # OBI = (average bid quantity - average ask quantity)
    #       / (average bid quantity + average ask quantity)
    total_volume = bid_vol + ask_vol
    obi = (
        (bid_vol - ask_vol)
        .div(total_volume.where(total_volume != 0))
        .fillna(0.0)
    )

    spread = ask_price - bid_price
    mid_price = (ask_price + bid_price) / 2

    # Build the extracted stream before resampling.
    feature_stream = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "OBI": obi,
            "MID_PRICE": mid_price,
            "SPREAD": spread,
            "ASK_PRICE": ask_price,
            "BID_PRICE": bid_price,
            "ASK_VOLUME": ask_vol,
            "BID_VOLUME": bid_vol,
            "local_timestamp": df["local_timestamp"],
        }
    )

    # Convert local capture timestamps from microseconds to UTC datetimes.
    feature_stream["local_datetime"] = pd.to_datetime(
        feature_stream["local_timestamp"],
        unit="us",
        utc=True,
        errors="coerce",
    )

    feature_stream = (
        feature_stream
        .dropna(subset=["local_datetime"])
        .set_index("local_datetime")
        .sort_index(kind="stable")
    )

    if feature_stream.empty:
        raise ValueError("No valid timestamps remain after timestamp conversion.")

    # Resample to exactly one row per 250 ms interval.
    # The last snapshot in each interval represents that interval.
    # Empty intervals are forward-filled to maintain a regular time series.
    feature_stream = (
        feature_stream
        .resample(RESAMPLE_INTERVAL)
        .last()
        .ffill()
    )

    feature_stream = feature_stream.reset_index()

    # Calculate the actual future return 4 resampled rows ahead (1 second).
    future_mid_price = feature_stream["MID_PRICE"].shift(
        -PREDICTION_HORIZON_ROWS
    )

    valid_prices = (
        feature_stream["MID_PRICE"].gt(0)
        & future_mid_price.gt(0)
        & feature_stream["MID_PRICE"].notna()
        & future_mid_price.notna()
    )

    feature_stream["FUTURE_RETURN"] = np.nan
    feature_stream.loc[valid_prices, "FUTURE_RETURN"] = np.log(
        future_mid_price.loc[valid_prices]
        / feature_stream.loc[valid_prices, "MID_PRICE"]
    )

    # Numeric target matching the existing model convention:
    # return > 0 => 1 (UP); return <= 0 => 0 (DOWN).
    # The final four rows have no future target and remain <NA>.
    direction = pd.Series(
        pd.NA,
        index=feature_stream.index,
        dtype="Int8",
    )
    direction.loc[valid_prices] = (
        feature_stream.loc[valid_prices, "FUTURE_RETURN"] > 0
    ).astype("int8")

    feature_stream["ACTUAL_DIRECTION"] = direction
    feature_stream["ACTUAL_DIRECTION_LABEL"] = (
        feature_stream["ACTUAL_DIRECTION"]
        .map({1: "UP", 0: "DOWN"})
        .fillna("UNKNOWN")
    )

    # CSV overview summaries are calculated from the resampled stream.
    components = {
        "ask": feature_stream["ASK_PRICE"],
        "bid": feature_stream["BID_PRICE"],
        "ask_volume": feature_stream["ASK_VOLUME"],
        "bid_volume": feature_stream["BID_VOLUME"],
    }

    orderbook_describe = pd.DataFrame(
        {
            name: values.describe()[
                ["mean", "std", "min", "25%", "50%", "75%", "max"]
            ]
            for name, values in components.items()
        }
    ).T

    feature_percentiles = pd.DataFrame(
        {
            "OBI": feature_stream["OBI"].describe(
                percentiles=PERCENTILES
            ),
            "SPREAD": feature_stream["SPREAD"].describe(
                percentiles=PERCENTILES
            ),
            "MID_PRICE": feature_stream["MID_PRICE"].describe(
                percentiles=PERCENTILES
            ),
        }
    ).drop(index=["count", "mean", "std"])

    return feature_percentiles, orderbook_describe, feature_stream


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Crypto Compass order-book features, resample to 250 ms, "
            "and calculate actual 1-second market directions."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV_PATH,
        help="Path to the Tardis Binance book_snapshot_25 CSV or CSV.GZ file.",
    )
    parser.add_argument(
        "--output-dir",
        default="csv_output",
        help="Directory for the generated CSV files (default: csv_output).",
    )
    args = parser.parse_args()

    feature_percentiles, orderbook_describe, feature_stream = summarize_csv(
        args.csv_path
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_percentiles.to_csv(
        output_dir / "feature_percentiles.csv",
        index=True,
    )
    orderbook_describe.to_csv(
        output_dir / "orderbook_describe.csv",
        index=True,
    )
    feature_stream.to_csv(
        output_dir / "feature_stream_250ms.csv",
        index=False,
    )

    # Export only the raw OBI values as a separate model-input stream.
    feature_stream[["OBI"]].to_csv(
        output_dir / "obi_stream_250ms.csv",
        index=False,
    )

    # Export timestamps, actual returns and actual directions for evaluation.
    feature_stream[
        [
            "local_datetime",
            "FUTURE_RETURN",
            "ACTUAL_DIRECTION",
            "ACTUAL_DIRECTION_LABEL",
        ]
    ].to_csv(
        output_dir / "actual_directions_250ms.csv",
        index=False,
    )



if __name__ == "__main__":
    main()


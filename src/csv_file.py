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
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract and summarize a Tardis Binance book_snapshot_25 CSV in chunks.

    Raw CSV rows are processed incrementally to reduce peak memory usage.
    The returned feature stream and summary tables retain the same columns
    and definitions as the original implementation.

    Input rows must be chronological by local_timestamp. This is checked
    across chunk boundaries because a global sort would require loading the
    complete raw file into memory.
    """
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer.")

    csv_source = str(csv_path).strip()
    is_url = csv_source.startswith(("http://", "https://"))

    if is_url:
        compression = (
            "gzip"
            if csv_source.split("?", 1)[0].lower().endswith(".gz")
            else "infer"
        )
        reader = pd.read_csv(
            csv_source,
            compression=compression,
            chunksize=chunksize,
        )
    else:
        local_path = Path(csv_source)
        if not local_path.exists():
            raise FileNotFoundError(f"CSV file not found: {local_path}")
        reader = pd.read_csv(local_path, chunksize=chunksize)

    required_columns = ["timestamp", "local_timestamp"]
    for side in ("asks", "bids"):
        for level in range(LEVELS_USED):
            required_columns.extend(
                [
                    f"{side}[{level}].price",
                    f"{side}[{level}].amount",
                ]
            )

    bid_price_columns = [f"bids[{i}].price" for i in range(LEVELS_USED)]
    ask_price_columns = [f"asks[{i}].price" for i in range(LEVELS_USED)]
    bid_volume_columns = [f"bids[{i}].amount" for i in range(LEVELS_USED)]
    ask_volume_columns = [f"asks[{i}].amount" for i in range(LEVELS_USED)]
    numeric_columns = [
        f"{side}[{level}].{field}"
        for side in ("asks", "bids")
        for level in range(LEVELS_USED)
        for field in ("price", "amount")
    ]

    resampled_chunks = []
    previous_local_timestamp = None
    total_raw_rows = 0

    for chunk_number, chunk in enumerate(reader, start=1):
        if chunk.empty:
            continue

        missing = [column for column in required_columns if column not in chunk.columns]
        if missing:
            preview = ", ".join(missing[:8])
            more = " ..." if len(missing) > 8 else ""
            raise ValueError(
                "This CSV does not have the expected Tardis book_snapshot_25 "
                f"columns. Missing: {preview}{more}"
            )

        total_raw_rows += len(chunk)

        chunk[numeric_columns] = chunk[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        chunk["local_timestamp"] = pd.to_numeric(
            chunk["local_timestamp"], errors="coerce"
        )
        chunk = chunk.dropna(subset=["local_timestamp"]).copy()

        if chunk.empty:
            continue

        # Preserve the original stable sort semantics while verifying that
        # chunk boundaries are globally chronological.
        chunk = chunk.sort_values("local_timestamp", kind="stable").reset_index(drop=True)
        first_timestamp = chunk["local_timestamp"].iloc[0]
        last_timestamp = chunk["local_timestamp"].iloc[-1]

        if (
            previous_local_timestamp is not None
            and first_timestamp < previous_local_timestamp
        ):
            raise ValueError(
                "The CSV is not globally chronological by local_timestamp. "
                "Chunked processing requires chronological input; sort the "
                "source CSV by local_timestamp first."
            )
        previous_local_timestamp = last_timestamp

        bid_price = chunk[bid_price_columns].mean(axis=1)
        ask_price = chunk[ask_price_columns].mean(axis=1)
        bid_vol = chunk[bid_volume_columns].mean(axis=1)
        ask_vol = chunk[ask_volume_columns].mean(axis=1)

        total_volume = bid_vol + ask_vol
        obi = (
            (bid_vol - ask_vol)
            .div(total_volume.where(total_volume != 0))
            .fillna(0.0)
        )

        feature_chunk = pd.DataFrame(
            {
                "timestamp": chunk["timestamp"],
                "OBI": obi,
                "MID_PRICE": (ask_price + bid_price) / 2,
                "SPREAD": ask_price - bid_price,
                "ASK_PRICE": ask_price,
                "BID_PRICE": bid_price,
                "ASK_VOLUME": ask_vol,
                "BID_VOLUME": bid_vol,
                "local_timestamp": chunk["local_timestamp"],
            }
        )

        feature_chunk["local_datetime"] = pd.to_datetime(
            feature_chunk["local_timestamp"],
            unit="us",
            utc=True,
            errors="coerce",
        )
        feature_chunk = (
            feature_chunk
            .dropna(subset=["local_datetime"])
            .set_index("local_datetime")
            .sort_index(kind="stable")
        )

        if feature_chunk.empty:
            continue

        # Resampling each chronological chunk is safe because the final
        # concatenation resolves bins split across chunk boundaries by
        # keeping the last observation for each 250 ms timestamp.
        resampled_chunk = (
            feature_chunk
            .resample(RESAMPLE_INTERVAL)
            .last()
        )
        resampled_chunks.append(resampled_chunk)

    if total_raw_rows == 0 or not resampled_chunks:
        raise ValueError("The input CSV contains no usable order-book rows.")

    feature_stream = pd.concat(resampled_chunks).sort_index(kind="stable")
    feature_stream = feature_stream[
        ~feature_stream.index.duplicated(keep="last")
    ].ffill()

    feature_stream = feature_stream.reset_index()

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



from data import load_file
import pandas as pd
import numpy as np

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

def feature_engineering():
    df = load_file()

    mean_bid_volume = df[
    [str(column) for column in range(3, 22, 2)]
    ].mean(axis=1)

    mean_ask_volume = df[
    [str(column) for column in range(23, 42, 2)]
    ].mean(axis=1)

    mean_bid_price = df[
    [str(column) for column in range(2, 21, 2)]
    ].mean(axis=1)

    mean_ask_price = df[
    [str(column) for column in range(22, 41, 2)]
    ].mean(axis=1)

# OBI 
    df['OBI'] = (mean_bid_volume - mean_ask_volume ) /  (mean_bid_volume + mean_ask_volume )

# Spread
    df["Spread"] = mean_ask_price - mean_bid_price

#Mid price 
    df['Mid_Price'] = (
        mean_ask_price + mean_bid_price
    ) / 2

# 1 - sec in future [1 - row = 0.25sec] 
    df['Mid_Price_t+h'] = (
        df['Mid_Price'].shift(-4)
    )

# Feature return 
    df['feature_return'] = np.log(
        df['Mid_Price_t+h']
        /
        df['Mid_Price']
    )
    df = df.dropna(subset=['feature_return'])

# Direction for UP | DOWN
    df['direction'] = (
        df['feature_return'] > 0
    ).astype('int8')

    n = len(df)

    train_end = int(n * 0.70)
    val1_end = int(n * 0.80)
    val2_end = int(n * 0.90)

#Split 
    train_df = df.iloc[:train_end].copy()

    val1_df = df.iloc[
        train_end:val1_end
    ].copy()

    val2_df = df.iloc[
        val1_end:val2_end
    ].copy()

    test_df = df.iloc[
        val2_end:
    ].copy()

    threshold = (
        train_df['feature_return']
    .abs()
    .quantile(0.75)
    )


    x_train = train_df['OBI']
    x_val1 = val1_df['OBI']
    x_val2 = val2_df['OBI']
    x_test = test_df['OBI']

    y_train = train_df['direction']
    y_val1 = val1_df['direction']
    y_val2 = val2_df['direction']
    y_test = test_df['direction']

    return (
        x_train,
        x_val1,
        x_val2,
        x_test,
        y_train,
        y_val1,
        y_val2,
        y_test,
        
    )

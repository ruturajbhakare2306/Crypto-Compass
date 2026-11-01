import streamlit as st
import pandas as pd
import numpy as np
from test_app import LivePredictionService
from csv_file import summarize_csv
from func import predict_obi_stream

st.set_page_config(layout="wide")


def get_live_service():
    if "live_prediction_service" not in st.session_state:
        st.session_state["live_prediction_service"] = LivePredictionService()
    return st.session_state["live_prediction_service"]


@st.fragment(run_every="250ms")
def render_block():
    if st.session_state.pop("live_control_full_rerun", False):
        st.rerun(scope="app")

    service = get_live_service()
    snapshot = service.snapshot()
    running = snapshot["running"]
    prediction = snapshot["prediction"]
    probabilities = snapshot["probabilities"]
    obi_values = snapshot["obi_values"]
    error = snapshot["error"]
    connection_status = snapshot["connection_status"]
    messages_received = snapshot["messages_received"]
    last_message_at = snapshot["last_message_at"]
    obi_count = snapshot["obi_count"]
    history_block = snapshot["prediction_history"]

    stop_column, resume_column = st.columns(2)
    with stop_column:
        st.button(
            "Stop prediction",
            key="stop_live_prediction",
            on_click=stop_live_prediction,
            disabled=not running,
        )
    with resume_column:
        st.button(
            "Resume prediction",
            key="resume_live_prediction",
            on_click=resume_live_prediction,
            disabled=running,
        )

    if not running:
        st.info("Live prediction is stopped.")

    status_col, messages_col, observations_col, last_message_col = st.columns(4)
    status_col.metric("WebSocket", connection_status)
    messages_col.metric("Messages received", messages_received)
    observations_col.metric("OBI collected", f"{obi_count}/12")
    last_message_col.metric("Last message (UTC)", last_message_at or "—")

    left, right = st.columns([7, 3])
    with left:
        top = st.container()
        top.subheader("Predictions")
        prediction_column, probability_column = top.columns(2)

        if prediction is None or probabilities is None:
            prediction_text = "Warming up..."
            probability_text = "Waiting for 12 OBI values"
            prediction_style = "gray"
        else:
            prediction_text = prediction
            prediction_style = "red" if prediction == "DOWN" else "green"
            predicted_probability = probabilities[0 if prediction == "DOWN" else 1]
            probability_text = f"{predicted_probability:.2%}"

        with prediction_column.container(border=True):
            st.markdown("**PREDICTION**")
            st.markdown(f"### :{prediction_style}[{prediction_text}]")

        with probability_column.container(border=True):
            st.markdown("**PREDICTED CLASS PROBABILITY**")
            st.markdown(f"### :{prediction_style}[{probability_text}]")

        bottom = st.container()
        bottom.subheader("OBI Sequences")
        padded_values = [None] * (12 - len(obi_values)) + obi_values

        for row_index in range(3):
            row = bottom.columns(4)
            for col, box in enumerate(row):
                index = row_index * 4 + col
                value = padded_values[index]
                with box.container(border=True):
                    st.markdown(f"**OBI : {index + 1}**")
                    if value is None:
                        st.markdown("### :gray[Waiting]")
                    else:
                        value_style = "green" if value > 0 else "red" if value < 0 else "gray"
                        st.markdown(f"### :{value_style}[{value:.4f}]")

        if error:
            st.caption(f"Live feed status: {error}")

    with right:
        history = right.container()
        history.subheader("PREDICTION HISTORY")

        for entry in history_block:
            time_stamp, prediction_col = history.columns([7,3])

            time_stamp.caption(entry['timestamp'])

            pred = entry['prediction']
            pred_style = (
                "red" if pred == "DOWN"
                else "green" if pred == "UP"
                else "gray"
            )

            prediction_col.markdown(
                f'**:{pred_style}[{pred}]**'
            )

            prediction_prob = entry['probabilities']
            predicted_probability_history = prediction_prob[0 if pred == "DOWN" else 1]

            prediction_col.caption('Predicted class Probability')
            prediction_col.markdown(f'### {predicted_probability_history:.2%}')

        if error:
            st.caption(f"Live feed status: {error}")

def line_chart(feature_stats):
    st.subheader("Order-book feature percentiles")
    percentiles = ["min", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "max"]
    chart_data = feature_stats.reindex(percentiles).rename_axis("Percentile").reset_index()
    chart_spec = {
    "layer": [
        {
            "mark": {"type": "line", "point": True, "color": "#1f77b4"},
            "encoding": {
                "x": {
                    "field": "Percentile",
                    "type": "ordinal",
                    "sort": percentiles,
                    "axis": {"title": "Percentile"},
                },
                "y": {
                    "field": "OBI",
                    "type": "quantitative",
                    "scale": {"domain": [-1, 1]},
                    "axis": {"title": "OBI", "orient": "left", "format": ".2f"},
                },
            },
        },
        {
            "mark": {"type": "line", "point": True, "color": "#ff7f0e"},
            "encoding": {
                "x": {"field": "Percentile",
                       "type": "ordinal", 
                       "sort": percentiles
                },
                "y": {
                    "field": "MID_PRICE",
                    "type": "quantitative",
                    "axis": {"title": "MID_PRICE", "orient": "right"},
                },
            },
        },
        {
            "mark": {"type": "line", "point": True, "color": "#2ca02c"},
            "encoding": {
                "x": {"field": "Percentile", 
                      "type": "ordinal", 
                      "sort": percentiles
                },
                "y": {
                    "field": "SPREAD",
                    "type": "quantitative",
                    "axis": {"title": "SPREAD", "orient": "right", "offset": 70},
                },
            },
        },
    ],
    "resolve": {"scale": {"y": "independent"}},
    "padding": {"right": 100},
    }
    st.vega_lite_chart(
    chart_data,
    chart_spec,
    alt="OBI from -1 to 1, mid-price, and spread by percentile, each with its own axis",
    )

def df_chart(orderbook_stats):
    st.subheader("Order-book stats from the data")
    st.dataframe(orderbook_stats)


def main_app_ui():
    st.title("CRYPTO COMPASS :compass:")
    st.toggle(
        "Predict live",
        key="predict_live",
        on_change=toggle_live_prediction,
    )
    page = st.session_state.get("app_page", "home")

    if page == "home":
        st.write("Visualize, analyze, and predict cryptocurrency market data.")
        with st.form("overview_form"):
            st.text_input(
                "CSV file URL or path",
                key="csv_file_url",
                placeholder="Enter a CSV file URL or path",
            )
            overview_submitted = st.form_submit_button("Overview")

        if overview_submitted:
            csv_source = st.session_state.get("csv_file_url", "").strip()
            if (
                len(csv_source) >= 2
                and csv_source[0] == csv_source[-1]
                and csv_source[0] in {"'", '"'}
            ):
                csv_source = csv_source[1:-1].strip()

            if not csv_source:
                st.error("Enter a CSV file URL or path first.")
            else:
                try:
                    feature_stats, orderbook_stats, feature_stream = summarize_csv(csv_source)
                    prediction_stream = predict_obi_stream(feature_stream["OBI"])
                    feature_stream = feature_stream.join(prediction_stream)
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Could not process the CSV or run Transformer predictions: {exc}")
                else:
                    st.session_state["feature_stats"] = feature_stats
                    st.session_state["orderbook_stats"] = orderbook_stats
                    st.session_state["feature_stream"] = feature_stream
                    st.session_state["app_page"] = "overview"
                    st.rerun()

    elif page == "overview":
        if st.button("Back", key="back_to_home"):
            st.session_state["app_page"] = "home"
            st.rerun()

        overview_tab, prediction_tab = st.tabs(["Overview", "OBI Predictions"])

        with overview_tab:
            df_chart(st.session_state["orderbook_stats"])
            line_chart(st.session_state["feature_stats"])
            st.subheader("Feature stream")
            st.dataframe(st.session_state["feature_stream"])

        with prediction_tab:
            st.subheader("Transformer predictions vs actual market direction")
            prediction_data = (
                    st.session_state["feature_stream"]
                    .iloc[11:]
                    .copy()
                    .reset_index(drop=True)
                )

            # Normalize the actual direction into UP / DOWN / UNKNOWN labels
            # so it can be compared directly with the model's prediction.
            if "ACTUAL_DIRECTION_LABEL" in prediction_data.columns:
                actual_labels = prediction_data["ACTUAL_DIRECTION_LABEL"].astype("string")
            elif "ACTUAL_DIRECTION" in prediction_data.columns:
                actual_values = prediction_data["ACTUAL_DIRECTION"]
                actual_labels = actual_values.map({
                    1: "UP",
                    0: "DOWN",
                    "1": "UP",
                    "0": "DOWN",
                    "UP": "UP",
                    "DOWN": "DOWN",
                }).fillna("UNKNOWN").astype("string")
            else:
                actual_labels = None

            comparison_data = pd.DataFrame(index=prediction_data.index)
            for column in ["local_datetime", "OBI", "MODEL_PREDICTION",
                           "DOWN_PROBABILITY", "UP_PROBABILITY"]:
                if column in prediction_data.columns:
                    comparison_data[column] = prediction_data[column]

            if actual_labels is not None:
                comparison_data["ACTUAL_DIRECTION"] = actual_labels

                if "MODEL_PREDICTION" in comparison_data.columns:
                    has_both = (
                        comparison_data["MODEL_PREDICTION"].notna()
                        & comparison_data["MODEL_PREDICTION"].isin(["UP", "DOWN"])
                        & comparison_data["ACTUAL_DIRECTION"].isin(["UP", "DOWN"])
                    )
                    comparison_data["RESULT"] = "NOT AVAILABLE"
                    comparison_data.loc[has_both, "RESULT"] = np.where(
                        comparison_data.loc[has_both, "MODEL_PREDICTION"]
                        == comparison_data.loc[has_both, "ACTUAL_DIRECTION"],
                        "CORRECT",
                        "INCORRECT",
                    )

            if "FUTURE_RETURN" in prediction_data.columns:
                comparison_data["FUTURE_RETURN"] = prediction_data["FUTURE_RETURN"]

            st.dataframe(
                comparison_data,
                use_container_width=True,
            )

    elif page == "live":
        render_block()


def toggle_live_prediction():
    if st.session_state["predict_live"]:
        get_live_service().start(reset=True)
        st.session_state["app_page"] = "live"
    else:
        get_live_service().stop()
        st.session_state["app_page"] = "home"


def stop_live_prediction():
    get_live_service().stop()
    st.session_state["predict_live"] = False
    st.session_state["live_control_full_rerun"] = True


def resume_live_prediction():
    get_live_service().start(reset=True)
    st.session_state["predict_live"] = True
    st.session_state["app_page"] = "live"
    st.session_state["live_control_full_rerun"] = True


main_app_ui()
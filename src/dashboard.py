from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DB_PATH, NUMERIC_METRICS
from src.detectors import load_joined_data
from src.evaluate import evaluate_method, load_latest_run_detections
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=30_000, key="datarefresh")
st.set_page_config(page_title="DriftWatch", layout="wide")
GAP_MINUTES = 5  # compress timeline gaps longer than this

st.title("DriftWatch — Laptop Health Anomaly Detection")


@st.cache_data
def get_joined_data(db_path: str) -> pd.DataFrame:
    try:
        df = load_joined_data(Path(db_path))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.error(f"Failed to load data from {db_path}: {e}")
        return pd.DataFrame()


@st.cache_data
def get_detections(db_path: str, method: str) -> pd.DataFrame:
    try:
        return load_latest_run_detections(Path(db_path), method)
    except Exception:
        # no run for this method yet — not fatal, just nothing to overlay
        return pd.DataFrame()


df = get_joined_data(str(DB_PATH))
if df.empty:
    st.warning("No data loaded — check DB_PATH and that detectors have been run.")
    st.stop()

# Data shown in the timeline
view = df.copy()

st.sidebar.header("Controls")
default_metric = "cpu_percent" if "cpu_percent" in NUMERIC_METRICS else NUMERIC_METRICS[0]
metric = st.sidebar.selectbox("Metric", NUMERIC_METRICS, index=NUMERIC_METRICS.index(default_metric))
methods = st.sidebar.multiselect(
    "Overlay detectors",
    ["zscore", "mahalanobis", "cusum", "ocsvm"],
    default=["cusum", "mahalanobis", "ocsvm"],
)

# --- Time series with flags ---
st.subheader(f"{metric} over time")

plot_df = view[["timestamp", metric]].copy().sort_values("timestamp")
plot_df["plot_time"] = plot_df["timestamp"]

offset = pd.Timedelta(0)
timestamps = plot_df["timestamp"].tolist()

for i in range(1, len(plot_df)):
    gap = timestamps[i] - timestamps[i - 1]
    if gap > pd.Timedelta(minutes=GAP_MINUTES):
        offset += gap
    plot_df.loc[plot_df.index[i], "plot_time"] -= offset

time_map = plot_df.set_index("timestamp")["plot_time"]

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=plot_df["plot_time"],
        y=plot_df[metric],
        mode="lines",
        name=metric,
        line=dict(color="steelblue"),
    )
)

marker_colors = {
    "zscore": "black",
    "mahalanobis": "green",
    "cusum": "red",
    "ocsvm": "orange",
}

for method in methods:
    detections = get_detections(str(db_path), method)
    if detections.empty:
        continue

    flagged_ids = detections.loc[detections["is_anomaly"], "metrics_id"]
    flagged_rows = view[view["metrics_id"].isin(flagged_ids)].copy()

    fig.add_trace(
        go.Scatter(
            x=flagged_rows["timestamp"].map(time_map),
            y=flagged_rows[metric],
            mode="markers",
            name=f"{method} alert",
            marker=dict(
                symbol="x",
                color=marker_colors.get(method, "black"),
                size=10,
            ),
        )
    )

fig.update_layout(
    height=450,
    xaxis_title="Time",
    yaxis_title=metric,
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)



# --- Detector performance table ---
st.subheader("Detector performance (latest runs)")
perf_rows = []
for method in ["zscore", "mahalanobis", "cusum", "ocsvm"]:
    try:
        perf_rows.append(evaluate_method(DB_PATH, method))
    except Exception as e:
        st.caption(f"Skipped '{method}': {e}")
if perf_rows:
    st.dataframe(pd.DataFrame(perf_rows).set_index("method"))
else:
    st.info("No evaluated detector runs found yet.")


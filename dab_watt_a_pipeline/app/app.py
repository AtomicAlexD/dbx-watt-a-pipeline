import os

import altair as alt
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

st.set_page_config(page_title="Turbine Health", layout="wide")
st.title("⚡ Turbine Health")
st.caption("Gold summary stats and anomalies, plus Silver's audit flags — which turbines are flaky.")

WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
assert WAREHOUSE_ID, "DATABRICKS_WAREHOUSE_ID must be set in app.yaml"

SILVER_TABLE = os.getenv("SILVER_TABLE", "silver.turbine_data.silver_turbine_readings")
GOLD_SUMMARY_TABLE = os.getenv("GOLD_SUMMARY_TABLE", "gold.turbine_data.daily_turbine_summary")
GOLD_ANOMALIES_TABLE = os.getenv("GOLD_ANOMALIES_TABLE", "gold.turbine_data.turbine_anomalies")

cfg = Config()


@st.cache_data(ttl=300)
def run_query(query: str) -> pd.DataFrame:
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: cfg.authenticate,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


turbine_ids = run_query(f"select distinct turbine_id from {GOLD_SUMMARY_TABLE} order by turbine_id")[
    "turbine_id"
].tolist()

# ---------------------------------------------------------------------------
# Turbine detail: the actual Gold output the brief asked for — daily
# min/max/avg power output, with anomalous real readings overlaid directly
# on the band so it's visible *why* a point was flagged, not just a count.
# ---------------------------------------------------------------------------

st.subheader("📈 Daily power output")
selected_turbine = st.selectbox("Turbine", turbine_ids)

summary_df = run_query(f"""
    select date, min_power_output, max_power_output, avg_power_output, reading_count, imputed_count
    from {GOLD_SUMMARY_TABLE}
    where turbine_id = {selected_turbine}
    order by date
""")

anomalies_df = run_query(f"""
    select date, timestamp, power_output
    from {GOLD_ANOMALIES_TABLE}
    where turbine_id = {selected_turbine} and is_anomaly = true
    order by timestamp
""")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Days of data", len(summary_df))
m2.metric("Total imputed readings", int(summary_df["imputed_count"].sum()))
m3.metric("Anomalous readings", len(anomalies_df))
m4.metric("Avg power output", f"{summary_df['avg_power_output'].mean():.2f}")

band = (
    alt.Chart(summary_df)
    .mark_area(opacity=0.25, color="#1f77b4")
    .encode(x=alt.X("date:T", title="Date"), y=alt.Y("min_power_output:Q", title="Power output"), y2="max_power_output:Q")
)
avg_line = (
    alt.Chart(summary_df)
    .mark_line(color="#1f77b4", strokeWidth=2)
    .encode(x="date:T", y="avg_power_output:Q")
)
anomaly_points = (
    alt.Chart(anomalies_df)
    .mark_point(color="red", size=80, filled=True)
    .encode(
        x="date:T",
        y="power_output:Q",
        tooltip=["timestamp:T", "power_output:Q"],
    )
    if not anomalies_df.empty
    else alt.Chart(pd.DataFrame({"date": [], "power_output": []})).mark_point()
)

st.altair_chart((band + avg_line + anomaly_points).properties(height=350).interactive(), use_container_width=True)
st.caption("Shaded band = daily min/max range. Line = daily average. Red points = flagged anomalies (>2σ from that day's mean).")

with st.expander("Daily summary data"):
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Fleet-wide data quality view
# ---------------------------------------------------------------------------

st.subheader("🩺 Fleet-wide data quality")

issues_df = run_query(f"""
    select
        turbine_id,
        sum(int(power_output_was_missing or wind_speed_was_missing or wind_direction_was_missing)) as missing_count,
        sum(int(power_output_was_invalid or wind_speed_was_invalid or wind_direction_was_invalid)) as invalid_count,
        sum(int(power_output_was_imputed)) as imputed_count,
        count(*) as total_readings
    from {SILVER_TABLE}
    group by turbine_id
    order by turbine_id
""")

fleet_anomalies_df = run_query(f"""
    select turbine_id, count(*) as anomaly_count
    from {GOLD_ANOMALIES_TABLE}
    where is_anomaly = true
    group by turbine_id
    order by turbine_id
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Data quality issues per turbine**")
    st.bar_chart(issues_df.set_index("turbine_id")[["missing_count", "invalid_count", "imputed_count"]])

with col2:
    st.markdown("**Anomalous readings per turbine**")
    if fleet_anomalies_df.empty:
        st.info("No anomalies flagged.")
    else:
        st.bar_chart(fleet_anomalies_df.set_index("turbine_id")["anomaly_count"])

worst = issues_df.assign(
    issue_rate=(issues_df["missing_count"] + issues_df["invalid_count"]) / issues_df["total_readings"]
).sort_values("issue_rate", ascending=False)
st.markdown("**🚨 Flakiest turbines**")
st.dataframe(
    worst[["turbine_id", "issue_rate", "missing_count", "invalid_count", "total_readings"]].head(5),
    use_container_width=True,
    hide_index=True,
)
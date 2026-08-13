import os

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

st.set_page_config(page_title="Turbine Health", layout="wide")
st.title("⚡ Turbine Health")
st.caption("Reads Silver's audit flags and Gold's anomaly counts — which turbines are flaky.")

WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
assert WAREHOUSE_ID, "DATABRICKS_WAREHOUSE_ID must be set in app.yaml"

SILVER_TABLE = os.getenv("SILVER_TABLE", "silver.turbine_data.silver_turbine_readings")
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

anomalies_df = run_query(f"""
    select turbine_id, count(*) as anomaly_count
    from {GOLD_ANOMALIES_TABLE}
    where is_anomaly = true
    group by turbine_id
    order by turbine_id
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data quality issues per turbine")
    st.bar_chart(issues_df.set_index("turbine_id")[["missing_count", "invalid_count", "imputed_count"]])
    st.dataframe(issues_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Anomalous readings per turbine")
    if anomalies_df.empty:
        st.info("No anomalies flagged.")
    else:
        st.bar_chart(anomalies_df.set_index("turbine_id")["anomaly_count"])
        st.dataframe(anomalies_df, use_container_width=True, hide_index=True)

st.divider()
worst = issues_df.assign(
    issue_rate=(issues_df["missing_count"] + issues_df["invalid_count"]) / issues_df["total_readings"]
).sort_values("issue_rate", ascending=False)
st.subheader("🚨 Flakiest turbines")
st.dataframe(
    worst[["turbine_id", "issue_rate", "missing_count", "invalid_count", "total_readings"]].head(5),
    use_container_width=True,
    hide_index=True,
)

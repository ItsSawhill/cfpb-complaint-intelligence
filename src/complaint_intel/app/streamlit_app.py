from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def main():
    st.set_page_config(page_title="CFPB Complaint Intelligence", layout="wide")
    st.title("CFPB Complaint Intelligence System")
    st.caption("Classification + Risk Scoring for complaint triage")

    df = load_data("data/processed/dashboard.parquet")

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date filter
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
        min_d, max_d = df["date_received"].min(), df["date_received"].max()
        start, end = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
        df = df[(df["date_received"] >= pd.to_datetime(start)) & (df["date_received"] <= pd.to_datetime(end))]

    # Risk level
    if "risk_level_final" in df.columns:
        levels = ["Low", "Medium", "High"]
        selected = st.sidebar.multiselect("Risk level", levels, default=levels)
        df = df[df["risk_level_final"].isin(selected)]

    # Product (predicted)
    if "predicted_product" in df.columns:
        products = sorted(df["predicted_product"].dropna().unique().tolist())
        pick = st.sidebar.multiselect("Predicted product", products, default=products[:10])
        if pick:
            df = df[df["predicted_product"].isin(pick)]

    # Company
    if "Company" in df.columns:
        companies = sorted(df["Company"].dropna().unique().tolist())
        comp = st.sidebar.multiselect("Company", companies, default=[])
        if comp:
            df = df[df["Company"].isin(comp)]

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Complaints", f"{len(df):,}")
    if "risk_score_final" in df.columns:
        c2.metric("Avg risk score", f"{df['risk_score_final'].dropna().mean():.1f}")
    if "prediction_confidence" in df.columns:
        c3.metric("Avg confidence", f"{df['prediction_confidence'].dropna().mean():.3f}")
    if "risk_level_final" in df.columns:
        c4.metric("High risk", f"{(df['risk_level_final']=='High').sum():,}")

    st.divider()

    # Charts
    left, right = st.columns(2)

    with left:
        st.subheader("Complaints over time")
        if "date_received" in df.columns:
            ts = df.groupby(pd.Grouper(key="date_received", freq="W")).size().reset_index(name="count")
            st.line_chart(ts.set_index("date_received")["count"])
        else:
            st.info("date_received not available")

    with right:
        st.subheader("Top predicted products")
        if "predicted_product" in df.columns:
            top = df["predicted_product"].value_counts().head(10).reset_index()
            top.columns = ["product", "count"]
            st.bar_chart(top.set_index("product")["count"])
        else:
            st.info("predicted_product not available")

    st.divider()

    # Triage table
    st.subheader("Triage Inbox (sorted by risk)")
    show_cols = [
        "complaint_id",
        "date_received",
        "Company",
        "State",
        "true_product",
        "predicted_product",
        "prediction_confidence",
        "risk_score_final",
        "risk_level_final",
        "risk_reasons_final",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    if "risk_score_final" in df.columns:
        df_view = df.sort_values("risk_score_final", ascending=False)
    else:
        df_view = df

    st.dataframe(df_view[show_cols].head(200), use_container_width=True)

    # Detail viewer
    st.subheader("Complaint Detail")
    cid = st.selectbox("Select complaint_id", df_view["complaint_id"].head(5000).tolist())
    row = df_view[df_view["complaint_id"] == cid].iloc[0]

    st.write("**Predicted product:**", row.get("predicted_product"))
    st.write("**Confidence:**", row.get("prediction_confidence"))
    st.write("**Risk score:**", row.get("risk_score_final"), "-", row.get("risk_level_final"))
    st.write("**Risk reasons:**", row.get("risk_reasons_final"))
    st.write("**Narrative:**")
    st.write(row.get("narrative", ""))


if __name__ == "__main__":
    main()

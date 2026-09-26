from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Customer Lifetime Value", page_icon="📈", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "clean" / "df_clean.csv"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load and prepare the cleaned transaction-level retail data."""
    df = pd.read_csv(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Amount"] = df["Quantity"] * df["Price"]
    df = df.dropna(subset=["Customer ID", "InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype("int64").astype(str)
    return df


def customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    summary = (
        df.groupby("Customer ID")
        .agg(
            Monetary=("Amount", "sum"),
            Frequency=("Invoice", "nunique"),
            Last_Purchase=("InvoiceDate", "max"),
            Total_Quantity=("Quantity", "sum"),
            Average_Order_Value=("Amount", "mean"),
            Country=("Country", "first"),
        )
        .reset_index()
    )
    summary["Recency_Days"] = (snapshot - summary["Last_Purchase"]).dt.days
    summary["CLV_Score"] = (
        summary["Monetary"].rank(pct=True) * 0.50
        + summary["Frequency"].rank(pct=True) * 0.30
        + (1 - summary["Recency_Days"].rank(pct=True)) * 0.20
    ) * 100
    summary["Segment"] = pd.qcut(
        summary["CLV_Score"].rank(method="first"),
        q=4,
        labels=["At risk", "Developing", "Loyal", "Champions"],
    )
    return summary.sort_values("CLV_Score", ascending=False)


try:
    transactions = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. Add `df_clean.csv` and refresh.")
    st.stop()

st.title("Customer Lifetime Value Dashboard")
st.caption("Interactive view of cleaned online retail transactions and customer value signals.")

min_date = transactions["InvoiceDate"].min().date()
max_date = transactions["InvoiceDate"].max().date()

with st.sidebar:
    st.header("Filters")
    selected_dates = st.date_input("Transaction date", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    countries = sorted(transactions["Country"].dropna().unique())
    selected_countries = st.multiselect("Countries", countries, default=countries)
    st.divider()
    st.caption("CLV Score is a relative 0–100 score based on monetary value (50%), purchase frequency (30%), and recency (20%).")

if len(selected_dates) != 2:
    st.info("Select a start and end date to view the dashboard.")
    st.stop()

start_date, end_date = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1]) + pd.Timedelta(days=1)
filtered = transactions.loc[
    (transactions["InvoiceDate"] >= start_date)
    & (transactions["InvoiceDate"] < end_date)
    & (transactions["Country"].isin(selected_countries))
].copy()

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

customers = customer_summary(filtered)
revenue = filtered["Amount"].sum()
orders = filtered["Invoice"].nunique()

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Revenue", f"£{revenue:,.0f}")
metric2.metric("Customers", f"{customers['Customer ID'].nunique():,}")
metric3.metric("Orders", f"{orders:,}")
metric4.metric("Average order value", f"£{revenue / orders:,.2f}" if orders else "—")

left, right = st.columns((1.15, 1))
with left:
    monthly = filtered.assign(Month=filtered["InvoiceDate"].dt.to_period("M").astype(str)).groupby("Month", as_index=False)["Amount"].sum()
    trend = px.line(monthly, x="Month", y="Amount", markers=True, title="Revenue trend", labels={"Amount": "Revenue (£)"})
    trend.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=340)
    st.plotly_chart(trend, use_container_width=True)

with right:
    segment_counts = customers["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
    segments = px.bar(segment_counts, x="Segment", y="Customers", color="Segment", title="Customer segments", category_orders={"Segment": ["At risk", "Developing", "Loyal", "Champions"]})
    segments.update_layout(showlegend=False, margin=dict(l=10, r=10, t=45, b=10), height=340)
    st.plotly_chart(segments, use_container_width=True)

st.subheader("Customer value map")
scatter = px.scatter(
    customers,
    x="Frequency",
    y="Monetary",
    size="Total_Quantity",
    color="Segment",
    hover_data={"Customer ID": True, "Recency_Days": True, "CLV_Score": ":.1f", "Average_Order_Value": ":.2f"},
    log_y=True,
    title="Customers by purchase frequency and spend",
    labels={"Monetary": "Total spend (£)", "Frequency": "Distinct orders"},
)
scatter.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=480)
st.plotly_chart(scatter, use_container_width=True)

st.subheader("Customer drill-down")
selected_customer = st.selectbox("Choose a customer", customers["Customer ID"].tolist())
profile = customers.loc[customers["Customer ID"] == selected_customer].iloc[0]
profile_cols = st.columns(5)
profile_cols[0].metric("Segment", str(profile["Segment"]))
profile_cols[1].metric("CLV Score", f"{profile['CLV_Score']:.1f}")
profile_cols[2].metric("Lifetime spend", f"£{profile['Monetary']:,.2f}")
profile_cols[3].metric("Purchase frequency", f"{profile['Frequency']:.0f}")
profile_cols[4].metric("Days since purchase", f"{profile['Recency_Days']:.0f}")

st.subheader("Ranked customer table")
display = customers[["Customer ID", "Country", "Segment", "CLV_Score", "Monetary", "Frequency", "Recency_Days", "Average_Order_Value"]].copy()
display = display.rename(columns={"Monetary": "Lifetime Spend (£)", "Frequency": "Orders", "Recency_Days": "Recency (days)", "Average_Order_Value": "Avg. line value (£)"})
st.dataframe(display, use_container_width=True, hide_index=True, column_config={
    "CLV_Score": st.column_config.NumberColumn(format="%.1f"),
    "Lifetime Spend (£)": st.column_config.NumberColumn(format="£%.2f"),
    "Avg. line value (£)": st.column_config.NumberColumn(format="£%.2f"),
})
st.download_button("Download customer summary (CSV)", display.to_csv(index=False).encode("utf-8"), "customer_clv_summary.csv", "text/csv")

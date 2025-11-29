# app.py
# ------------------------------------------------------------
# ACCURATE ADVANCED INCOME ANALYSIS DASHBOARD
# (Upload your dataset via the sidebar)
# ------------------------------------------------------------

from typing import Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="ACCURATE ADVANCED INCOME ANALYSIS DASHBOARD",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Sidebar: upload + filters
# ============================================================
st.sidebar.header("📂 Upload & Filters")

uploaded_file = st.sidebar.file_uploader("Upload earnings CSV file", type=["csv"])

if uploaded_file is None:
    st.sidebar.info("Upload a CSV file (earnings data) to begin.")
    st.warning("⚠ Please upload a CSV file from the sidebar to continue.")
    st.stop()

@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

# load dataframe
try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"❌ Failed to read CSV: {e}")
    st.stop()

st.sidebar.markdown(f"**Loaded file:** `{getattr(uploaded_file, 'name', 'uploaded_file')}`")
st.write("🎯 Loaded CSV Successfully")
st.write("Rows:", df.shape[0], "Columns:", df.shape[1])

# ============================================================
# Basic validation and cleaning
# ============================================================
# Expected columns (not all are required, but recommended)
expected_cols = [
    "date", "user_id", "platform", "orders_completed", "hours_worked",
    "gross_earning", "platform_commission", "tip_amount", "fuel_cost_deducted",
    "net_earning", "festival_day"
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.warning("⚠ The following recommended columns are missing from your file: " + ", ".join(missing))
    st.info("The app will continue with available columns, but some visuals/KPIs may not show as expected.")

# convert and coerce
if "date" in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
else:
    st.error("❌ `date` column is required for time-based analysis. Please include it in the CSV.")
    st.stop()

# Convert numeric columns if present
numeric_columns_to_try = [
    "orders_completed", "hours_worked", "gross_earning",
    "platform_commission", "tip_amount", "fuel_cost_deducted",
    "net_earning", "festival_day"
]

for col in numeric_columns_to_try:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows without net_earning if the column exists
if "net_earning" in df.columns:
    df = df.dropna(subset=["net_earning"])
else:
    st.error("❌ `net_earning` column is required. Please include it in the CSV.")
    st.stop()

df = df.sort_values("date").reset_index(drop=True)

# Derived columns (safe)
df['weekday'] = df['date'].dt.day_name()
df['week'] = df['date'].dt.isocalendar().week
df['month'] = df['date'].dt.month_name()

# earnings_per_hour and earnings_per_order with safe division
if "hours_worked" in df.columns:
    df['earnings_per_hour'] = df['net_earning'] / df['hours_worked'].replace({0: np.nan})
else:
    df['earnings_per_hour'] = np.nan

if "orders_completed" in df.columns:
    df['earnings_per_order'] = df['net_earning'] / df['orders_completed'].replace({0: np.nan})
else:
    df['earnings_per_order'] = np.nan

# ============================================================
# Sidebar: interactive filters (date range + platform + search)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

# Prepare min/max dates for defaults
min_date = df['date'].min()
max_date = df['date'].max()

# Use Any typing for st.date_input return to avoid type-linter confusion
date_input_value: Any = st.sidebar.date_input("Select date range", [min_date.date(), max_date.date()])

# Normalize the return value of st.date_input which can be:
# - a single date (datetime.date)
# - a list/tuple of two dates [start, end]
try:
    if isinstance(date_input_value, (list, tuple)):
        # ensure we have two items; if not, fall back to min/max
        if len(date_input_value) >= 2:
            start_date = pd.to_datetime(date_input_value[0])
            end_date = pd.to_datetime(date_input_value[1])
        elif len(date_input_value) == 1:
            start_date = pd.to_datetime(date_input_value[0])
            end_date = start_date
        else:
            start_date = pd.to_datetime(min_date)
            end_date = pd.to_datetime(max_date)
    else:
        # single date selected
        start_date = pd.to_datetime(date_input_value)
        end_date = start_date
except Exception:
    # fallback safe range
    start_date = pd.to_datetime(min_date)
    end_date = pd.to_datetime(max_date)

# platform filter if available
platforms = df['platform'].dropna().unique().tolist() if 'platform' in df.columns else []
selected_platforms = st.sidebar.multiselect("Platform (select one or more)", options=platforms, default=platforms)

# user search
user_search = None
if 'user_id' in df.columns:
    user_list = df['user_id'].dropna().unique().tolist()
    user_search = st.sidebar.selectbox("Filter by user (optional)", options=["All"] + user_list)
else:
    st.sidebar.info("No `user_id` column found; skipping user filter.")

# apply filters
df_filtered = df.copy()

# apply date filter using normalized start_date/end_date
try:
    df_filtered = df_filtered[(df_filtered['date'] >= start_date) & (df_filtered['date'] <= end_date)]
except Exception:
    st.error("Invalid date range selected. Showing full range.")
    df_filtered = df.copy()

# platform filter
if platforms:
    if selected_platforms:
        df_filtered = df_filtered[df_filtered['platform'].isin(selected_platforms)]

# user filter
if user_search and user_search != "All":
    df_filtered = df_filtered[df_filtered['user_id'] == user_search]

if df_filtered.empty:
    st.warning("No data after applying filters. Try changing the filters.")
    st.stop()

# ============================================================
# 3) KPIs
# ============================================================
st.markdown("---")
st.header("Key Performance Indicators")

# total_net
total_net = df_filtered['net_earning'].sum()
# avg_daily
avg_daily = df_filtered.groupby("date")['net_earning'].sum().mean()
# best day
daily_group = df_filtered.groupby("date")['net_earning'].sum()
if not daily_group.empty:
    best_day = daily_group.idxmax()
    best_day_amt = daily_group.max()
else:
    best_day = None
    best_day_amt = 0

st.markdown(f"""
<div style="display:flex;gap:20px;margin:20px 0;">
    <div style="padding:12px;background:#2E86C1;color:white;border-radius:8px;">
        <h3>Total Net Earnings</h3><h2>₹{total_net:,.0f}</h2>
    </div>
    <div style="padding:12px;background:#1ABC9C;color:white;border-radius:8px;">
        <h3>Average Daily</h3><h2>₹{avg_daily:,.0f}</h2>
    </div>
    <div style="padding:12px;background:#F39C12;color:white;border-radius:8px;">
        <h3>Best Day</h3><h2>{pd.Timestamp(best_day).date() if best_day is not None else 'N/A'} → ₹{best_day_amt:,.0f}</h2>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 4) DAILY TREND + MOVING AVERAGE
# ============================================================
st.markdown("---")
st.subheader("📅 Daily Net Earnings + Moving Average")

daily = df_filtered.groupby("date")['net_earning'].sum().reset_index().sort_values("date")
daily['7d_avg'] = daily['net_earning'].rolling(7, min_periods=1).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily['date'], y=daily['net_earning'],
    mode="lines+markers", name="Daily Earnings",
    line=dict(color="#3498DB")
))
fig.add_trace(go.Scatter(
    x=daily['date'], y=daily['7d_avg'],
    mode="lines", name="7-Day Avg",
    line=dict(color="#E74C3C", width=3)
))
fig.update_layout(xaxis_title="Date", yaxis_title="Net Earnings (₹)", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5) PLATFORM-WISE TREND
# ============================================================
if 'platform' in df_filtered.columns:
    st.markdown("---")
    st.subheader("🚚 Platform-wise Earnings Trend")
    plat_daily = df_filtered.groupby(['date', 'platform'])['net_earning'].sum().reset_index()
    fig = px.line(
        plat_daily, x='date', y='net_earning',
        color='platform', markers=True,
        title="Platform-wise Earnings Trend"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Net Earnings (₹)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6) HEATMAP: WEEKDAY × MONTH
# ============================================================
st.markdown("---")
st.subheader("🔥 Heatmap: Weekday × Month Earnings")
heatmap = df_filtered.pivot_table(
    index='weekday', columns='month', values='net_earning', aggfunc='sum'
).reindex(index=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

if heatmap.isnull().all().all():
    st.info("No data available to build heatmap (missing weekday/month values).")
else:
    fig = px.imshow(
        heatmap.fillna(0),
        aspect="auto",
        title="Heatmap: Weekday × Month Earnings",
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Day")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 7) DISTRIBUTION OF EARNINGS
# ============================================================
st.markdown("---")
st.subheader("📊 Distribution of Net Earnings")
fig = px.histogram(
    df_filtered, x='net_earning', nbins=30,
    title="Distribution of Net Earnings"
)
st.plotly_chart(fig, use_container_width=True)

fig_box = px.box(df_filtered, y='net_earning', points="all", title="📦 Earnings Outlier Boxplot")
st.plotly_chart(fig_box, use_container_width=True)

# ============================================================
# 8) OUTLIER DETECTION (Z-SCORE)
# ============================================================
st.markdown("---")
st.subheader("⚠ Outlier Detection (Z ≥ 2)")
daily2 = daily.copy()
daily2['z'] = (daily2['net_earning'] - daily2['net_earning'].mean()) / (daily2['net_earning'].std(ddof=0) if daily2['net_earning'].std(ddof=0) != 0 else np.nan)
outliers = daily2[daily2['z'] >= 2]

fig_out = px.scatter(
    daily2, x='date', y='net_earning',
    color=(daily2['z'] >= 2),
    labels={'color': 'Outlier'},
    title="Outlier Detection (Z ≥ 2)"
)
st.plotly_chart(fig_out, use_container_width=True)

st.write("⚠ Outlier Days:")
if outliers.empty:
    st.write("No outlier days found with Z ≥ 2.")
else:
    st.dataframe(outliers[['date', 'net_earning', 'z']].reset_index(drop=True))

# ============================================================
# 9) EARNINGS PER HOUR
# ============================================================
st.markdown("---")
st.subheader("⏳ Earnings per Hour vs Hours Worked")
if 'hours_worked' in df_filtered.columns:
    fig = px.scatter(
        df_filtered,
        x='hours_worked', y='earnings_per_hour',
        size='net_earning' if 'net_earning' in df_filtered.columns else None,
        color='platform' if 'platform' in df_filtered.columns else None,
        hover_data=[col for col in ['user_id', 'orders_completed'] if col in df_filtered.columns],
        title="Earnings per Hour vs Hours Worked"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("`hours_worked` column not found; cannot show Earnings per Hour chart.")

# ============================================================
# 10) PLATFORM CONTRIBUTION
# ============================================================
st.markdown("---")
st.subheader("🍕 Platform-wise Contribution")
if 'platform' in df_filtered.columns:
    platform_summary = df_filtered.groupby('platform')['net_earning'].sum().reset_index()
    fig = px.pie(
        platform_summary, names='platform', values='net_earning',
        title="Platform-wise Contribution",
        hole=0.45
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("`platform` column not found; skipping platform contribution pie chart.")

# ============================================================
# 11) USER-WISE TABLE
# ============================================================
st.markdown("---")
st.subheader("👤 User-wise Earnings Summary")
if 'user_id' in df_filtered.columns:
    user_sum = df_filtered.groupby('user_id')['net_earning'].sum().reset_index().sort_values('net_earning', ascending=False)
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=['User','Total Net Earnings'], fill_color="#2E86C1", font=dict(color="white")),
        cells=dict(values=[user_sum['user_id'], user_sum['net_earning']],
                   fill_color="#D6EAF8")
    )])
    fig_table.update_layout(title="User-wise Earnings Summary")
    st.plotly_chart(fig_table, use_container_width=True)
else:
    st.info("`user_id` column not found; skipping user-wise table.")

# ============================================================
# Footer / Export
# ============================================================
st.markdown("---")
st.subheader("Download filtered data")
csv = df_filtered.to_csv(index=False)
st.download_button("⬇️ Download filtered CSV", csv, file_name="filtered_earnings.csv", mime="text/csv")

st.caption("Built with ❤️ — Upload different CSVs to analyze other datasets.")

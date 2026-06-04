import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("📊 Sales Analysis Dashboard")

# Load Data
df = pd.read_csv("sales_dataset_2026.csv")

df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])

df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['hour'] = df['time'].dt.hour
df['day'] = df['date'].dt.day_name()

# ---------------- CATEGORY ----------------
st.subheader("Profit Distribution by Category")

category = df.groupby('category')[['total','profit']].sum().sort_values(by='profit', ascending=False)

fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(category.index, category['profit'])

ax.set_title('Profit Distribution by Category')
ax.set_xlabel('Category')
ax.set_ylabel('Profit (₹)')
ax.tick_params(axis='x', rotation=30)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, yval,
            f'{yval/100000:.1f}L',
            ha='center')

st.pyplot(fig)

# ---------------- REGION + CATEGORY ----------------
st.subheader("Category Contribution Across Regions")

pivot = df.pivot_table(
    values='profit',
    index='region',
    columns='category',
    aggfunc='sum'
)

pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)*100

fig, ax = plt.subplots(figsize=(12,7))
pivot_pct.plot(kind='bar', stacked=True, ax=ax)

ax.set_title('Category Contribution (%) Across Regions')
ax.set_xlabel('Region')
ax.set_ylabel('Percentage Contribution (%)')

st.pyplot(fig)

# ---------------- HOUR ----------------
st.subheader("Sales Trend by Hour")

hourly = df.groupby('hour')['total'].sum()

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(hourly.index, hourly.values)

ax.set_title('Sales Trend by Hour')
ax.set_xlabel('Hour')
ax.set_ylabel('Sales (₹)')
ax.grid(True)

st.pyplot(fig)

# ---------------- DAY ----------------
st.subheader("Daily Sales Distribution")

day_sales = df.groupby('day')['total'].sum()

order = [
    'Monday','Tuesday','Wednesday',
    'Thursday','Friday','Saturday','Sunday'
]

day_sales = day_sales.reindex(order)

fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(day_sales.index, day_sales.values)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, yval,
            f'{yval/100000:.1f}L',
            ha='center')

ax.set_title('Daily Sales Distribution')
ax.set_xlabel('Day')
ax.set_ylabel('Sales (₹)')

st.pyplot(fig)

# ---------------- DISCOUNT ----------------
st.subheader("Average Profit by Discount Range")

df['discount_range'] = pd.cut(df['discount'], bins=5)

discount = df.groupby('discount_range')['profit'].mean()

fig, ax = plt.subplots(figsize=(10,6))
bars = ax.bar(discount.index.astype(str), discount.values)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, yval,
            f'{yval/1000:.1f}K',
            ha='center')

ax.set_title('Average Profit by Discount Range')
ax.set_xlabel('Discount Range')
ax.set_ylabel('Average Profit (₹)')
ax.tick_params(axis='x', rotation=30)

st.pyplot(fig)

# ---------------- FORECAST ----------------
st.subheader("Sales Forecast Trend")

daily_sales = df.groupby('date')['total'].sum()
daily_sales_ma = daily_sales.rolling(window=7).mean()

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(daily_sales.index,
        daily_sales.values,
        alpha=0.3,
        label='Actual')

ax.plot(daily_sales.index,
        daily_sales_ma,
        linewidth=3,
        label='7-Day Trend')

ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax.legend()
ax.grid(True)

ax.set_title('Sales Forecast Trend')
ax.set_xlabel('Date')
ax.set_ylabel('Sales (₹)')

plt.xticks(rotation=45)

st.pyplot(fig)

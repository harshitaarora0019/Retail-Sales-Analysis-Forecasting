import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("sales_dataset_2026.csv")

df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])

df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')

df['hour'] = df['time'].dt.hour
df['day'] = df['date'].dt.day_name()

# CATEGORY ANALYSIS
category = df.groupby('category')[['total','profit']].sum().sort_values(by='profit', ascending=False)

plt.figure(figsize=(10,6))
bars = plt.bar(category.index, category['profit'])
plt.title('Profit by Category')
plt.xticks(rotation=30)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, yval, f'{yval/100000:.1f}L', ha='center')

plt.tight_layout()
plt.show()
plt.close()

# REGION + CATEGORY
pivot = df.pivot_table(values='profit', index='region', columns='category', aggfunc='sum')
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)*100

pivot_pct.plot(kind='bar', stacked=True, figsize=(12,7))
plt.title('Category Contribution (%) by Region')
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
plt.close()

# HOURLY SALES
hourly = df.groupby('hour')['total'].sum()

plt.figure(figsize=(10,6))
plt.plot(hourly.index, hourly.values)
plt.title('Sales by Hour')
plt.grid(True)

plt.show()
plt.close()

# DAILY SALES BY DAY
day_sales = df.groupby('day')['total'].sum()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_sales = day_sales.reindex(order)

plt.figure(figsize=(10,6))
bars = plt.bar(day_sales.index, day_sales.values)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, yval, f'{yval/100000:.1f}L', ha='center')

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
plt.close()

# DISCOUNT ANALYSIS
df['discount_range'] = pd.cut(df['discount'], bins=5)
discount = df.groupby('discount_range')['profit'].mean()

plt.figure(figsize=(10,6))
bars = plt.bar(discount.index.astype(str), discount.values)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, yval, f'{yval/1000:.1f}k', ha='center')

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
plt.close()

# FORECASTING (MOVING AVERAGE)
daily_sales = df.groupby('date')['total'].sum()
daily_sales_ma = daily_sales.rolling(window=7).mean()

plt.figure(figsize=(12,6))
plt.plot(daily_sales.index, daily_sales.values, alpha=0.3, label='Actual')
plt.plot(daily_sales.index, daily_sales_ma, color='red', linewidth=3, label='Trend')

plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# =====================================
# Step 1: Generate Synthetic Dataset
# =====================================
def generate_synthetic_data(save_to_csv=False):
    np.random.seed(42)  # For reproducibility
    
    # Generate monthly dates from 2010 to 2023
    dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='M')
    
    # Define industries and regions
    industries = ['Technology', 'Healthcare', 'Manufacturing', 'Retail', 'Education']
    regions = ['North', 'South', 'East', 'West']
    
    # Simulate employment data with trends
    data = []
    for date in dates:
        for industry in industries:
            for region in regions:
                # Base employment with industry-specific trends
                base = {
                    'Technology': 10000,
                    'Healthcare': 8000,
                    'Manufacturing': 7000,
                    'Retail': 6000,
                    'Education': 5000
                }[industry]
                
                # Simulate growth over time
                years_passed = (date.year - 2010)
                growth = {
                    'Technology': 0.15 * years_passed,
                    'Healthcare': 0.08 * years_passed,
                    'Manufacturing': -0.05 * years_passed,
                    'Retail': 0.03 * years_passed,
                    'Education': 0.06 * years_passed
                }[industry]
                
                # Add regional variation and noise
                employment = base * (1 + growth) + np.random.randint(-500, 500)
                employment = max(employment, 1000)  # Ensure non-negative
                
                data.append({
                    'date': date,
                    'industry': industry,
                    'region': region,
                    'employment_count': int(employment)
                })
    
    df = pd.DataFrame(data)
    if save_to_csv:
        df.to_csv('synthetic_employment_data.csv', index=False)
    return df

# Generate and load data
df = generate_synthetic_data(save_to_csv=True)

# =====================================
# Step 2: Exploratory Data Analysis (EDA)
# =====================================
# Add year/month columns
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Plot total employment trends
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='year', y='employment_count', estimator='sum', ci=None)
plt.title('Total Employment Trends (2010–2023)')
plt.ylabel('Total Employment (Millions)')
plt.show()

# Industry-specific trends
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=df.groupby(['year', 'industry'])['employment_count'].sum().reset_index(),
    x='year', y='employment_count', hue='industry'
)
plt.title('Employment Growth by Industry')
plt.show()

# Regional heatmap
region_pivot = df.pivot_table(
    index='year', columns='region', values='employment_count', aggfunc='sum'
)
plt.figure(figsize=(12, 8))
sns.heatmap(region_pivot, annot=True, fmt=".0f", cmap='viridis')
plt.title('Regional Employment Distribution (Heatmap)')
plt.show()

# =====================================
# Step 3: Advanced Analysis
# =====================================
# YoY Growth Calculation
df_yoy = df.groupby(['industry', 'year'])['employment_count'].sum().reset_index()
df_yoy['yoy_growth'] = df_yoy.groupby('industry')['employment_count'].pct_change() * 100

# Plot YoY Growth
plt.figure(figsize=(14, 6))
sns.barplot(data=df_yoy, x='year', y='yoy_growth', hue='industry')
plt.title('Year-over-Year Growth by Industry')
plt.ylabel('Growth Rate (%)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Time Series Decomposition (Technology Sector)
tech_df = df[df['industry'] == 'Technology']
tech_df = tech_df.set_index('date').resample('Y').sum()
decomposition = seasonal_decompose(tech_df['employment_count'], model='additive', period=1)
decomposition.plot()
plt.suptitle('Technology Sector: Time Series Decomposition')
plt.tight_layout()
plt.show()

# =====================================
# Step 4: Simple Forecasting (Linear Regression)
# =====================================
from sklearn.linear_model import LinearRegression

# Prepare data
X = tech_df.reset_index()[['year']]
X['year'] = X['year'].dt.year - 2010  # Convert to numerical
y = tech_df['employment_count']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 5 years
future_years = np.array([14, 15, 16, 17, 18]).reshape(-1, 1)  # 2024–2028
predictions = model.predict(future_years)

print("Forecasted Employment (Technology Sector):")
for year, pred in zip(range(2024, 2029), predictions):
    print(f"{year}: {int(pred):,}")

import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
 
def summarize_data(df):
    summary = df.describe()
    return summary

def missing_value_report(df):
    missing_values = df.isna().sum()
    columns_with_missing = missing_values[missing_values > (0.05 * len(df))].index.tolist()
    return missing_values, columns_with_missing

def check_missing_values(df):
    return df.isna().sum()

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def clean_data(df, columns):
    for column in columns:
        # Fill missing values with median
        df[column].fillna(df[column].median(), inplace=True)
        # Detect outliers
        outliers = detect_outliers(df, column)
        # Remove outliers
        df = df[~df.index.isin(outliers.index)]
    return df

def compute_z_scores(df, columns):
    z_scores = zscore(df[columns], nan_policy='omit')
    return pd.DataFrame(z_scores, columns=columns, index=df.index)

def flag_outliers(z_scores):
    return (abs(z_scores) > 3).any(axis=1)  # Flag rows where any Z-score is > 3

def clean_data(df, columns):
    for column in columns:
        df[column].fillna(df[column].median()) # Fill missing values with median
    return df



## Line or bar charts of GHI, DNI, DHI, Tamb vs. Timestamp.
def plot_time_series(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    plt.figure(figsize=(14, 8))

    # Plotting GHI
    plt.subplot(4, 1, 1)
    plt.plot(df.index, df['GHI'], label='GHI', color='orange')
    plt.title('Global Horizontal Irradiance (GHI)')
    plt.xlabel('Timestamp')
    plt.ylabel('GHI (W/m²)')
    plt.legend()
    plt.grid()

    # Plotting DNI
    plt.subplot(4, 1, 2)
    plt.plot(df.index, df['DNI'], label='DNI', color='blue')
    plt.title('Direct Normal Irradiance (DNI)')
    plt.xlabel('Timestamp')
    plt.ylabel('DNI (W/m²)')
    plt.legend()
    plt.grid()

    # Plotting DHI
    plt.subplot(4, 1, 3)
    plt.plot(df.index, df['DHI'], label='DHI', color='green')
    plt.title('Diffuse Horizontal Irradiance (DHI)')
    plt.xlabel('Timestamp')
    plt.ylabel('DHI (W/m²)')
    plt.legend()
    plt.grid()

    # Plotting Tamb
    plt.subplot(4, 1, 4)
    plt.plot(df.index, df['Tamb'], label='Tamb', color='red')
    plt.title('Ambient Temperature (Tamb)')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


## Observe patterns by month, trends throughout day, or anomalies, such as peaks in solar irradiance or temperature fluctuation

def plot_monthly_patterns(df):
    monthly_data = df.resample('M').mean()
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_data.index, monthly_data['GHI'], label='Monthly Average GHI', color='orange')
    plt.plot(monthly_data.index, monthly_data['Tamb'], label='Monthly Average Temperature', color='red')
    plt.title('Monthly Patterns in GHI and Temperature')
    plt.xlabel('Month')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.show()

def plot_daily_trends(df):
    daily_data = df.resample('D').mean()
    plt.figure(figsize=(14, 6))
    plt.plot(daily_data.index, daily_data['GHI'], label='Daily Average GHI', color='orange')
    plt.plot(daily_data.index, daily_data['Tamb'], label='Daily Average Temperature', color='red')
    plt.title('Daily Trends in GHI and Temperature')
    plt.xlabel('Day')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.show()

def detect_anomalies(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['GHI'], label='GHI', color='orange')
    plt.title('GHI with Anomalies')
    plt.xlabel('Timestamp')
    plt.ylabel('GHI (W/m²)')
    
    # Detect peaks as anomalies
    peaks = df['GHI'][(df['GHI'] > df['GHI'].mean() + 3 * df['GHI'].std())]
    plt.scatter(peaks.index, peaks, color='red', label='Anomalies (Peaks)', zorder=5)
    
    plt.legend()
    plt.grid()
    plt.show()


## Group by Cleaning flag and plot average ModA & ModB pre/post-clean.
def plot_cleaning_impact(df):
    # Check for available columns
    print("Available columns:", df.columns)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Group by the cleaning flag and calculate the average for ModA and ModB
    cleaning_impact = df.groupby('Cleaning')[['ModA', 'ModB']].mean().reset_index()

    # Plotting
    cleaning_impact.plot(x='Cleaning', kind='bar', figsize=(10, 6))
    plt.title('Average ModA & ModB Pre/Post-Cleaning')
    plt.ylabel('Average Values')
    plt.xlabel('Cleaning Flag')
    plt.xticks(rotation=0)  # Rotate x labels for better visibility
    plt.grid(axis='y')
    plt.legend(title='Metrics')
    plt.show()




## Heatmap of correlations (GHI, DNI, DHI, TModA, TModB).


def plot_correlation_heatmap(df):
    # Select relevant columns
    columns_of_interest = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    correlation_matrix = df[columns_of_interest].corr()
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.show()


## ## Scatter plots: WS, WSgust, WD vs. GHI; RH vs. Tamb or RH vs. GHI.

def plot_scatter_plots(df):
    plt.figure(figsize=(14, 10))
    # Scatter plot: WS vs GHI
    plt.subplot(2, 2, 1)
    plt.scatter(df['WS'], df['GHI'], alpha=0.6)
    plt.title('Wind Speed (WS) vs. Global Horizontal Irradiance (GHI)')
    plt.xlabel('Wind Speed (WS)')
    plt.ylabel('GHI (W/m²)')
    plt.grid()

    # Scatter plot: WSgust vs GHI
    plt.subplot(2, 2, 2)
    plt.scatter(df['WSgust'], df['GHI'], alpha=0.6, color='orange')
    plt.title('Wind Gust (WSgust) vs. Global Horizontal Irradiance (GHI)')
    plt.xlabel('Wind Gust (WSgust)')
    plt.ylabel('GHI (W/m²)')
    plt.grid()

    # Scatter plot: WD vs GHI
    plt.subplot(2, 2, 3)
    plt.scatter(df['WD'], df['GHI'], alpha=0.6, color='green')
    plt.title('Wind Direction (WD) vs. Global Horizontal Irradiance (GHI)')
    plt.xlabel('Wind Direction (WD)')
    plt.ylabel('GHI (W/m²)')
    plt.grid()

    # Scatter plot: RH vs Tamb
    plt.subplot(2, 2, 4)
    plt.scatter(df['RH'], df['Tamb'], alpha=0.6, color='red')
    plt.title('Relative Humidity (RH) vs. Ambient Temperature (Tamb)')
    plt.xlabel('Relative Humidity (RH)')
    plt.ylabel('Ambient Temperature (Tamb)')
    plt.grid()

    plt.tight_layout()
    plt.show()



    ## Wind & Distribution Analysis (Wind rose or radial bar plot of WS/WD.)

def plot_wind_rose(df):
    """Create a wind rose plot from the DataFrame."""
    # Check for required columns
    if 'WD' not in df or 'WS' not in df:
        raise ValueError("DataFrame must contain 'Wind_Direction' and 'Wind_Speed' columns.")
    
    # Drop NaN values
    df = df.dropna(subset=['WD', 'WS'])

    # Prepare data
    wind_direction = df['WD'].values
    wind_speed = df['WS'].values

    # Create bins for wind direction (36 bins for 10-degree intervals)
    num_bins = 36
    direction_bins = np.linspace(0, 360, num_bins + 1)

    # Create histogram of wind speeds binned by wind direction
    wind_hist, _ = np.histogram(wind_direction, bins=direction_bins, weights=wind_speed)

    # Calculate the midpoints of the bins for plotting
    theta = np.deg2rad(direction_bins[:-1])  # Convert degrees to radians
    width = np.deg2rad(10)  # Width of each bin

    # Ensure the histogram has the same length as theta
    if len(wind_hist) != len(theta):
        raise ValueError("Wind histogram and theta lengths do not match.")

    # Create the wind rose plot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    ax.bar(theta, wind_hist, width=width, alpha=0.7, color='blue')
    ax.set_title('Wind Rose Plot', va='bottom')
    plt.show()




## Wind & Distribution Analysis (Histograms for GHI and one other variable (e.g. WS)).

def plot_histograms(df):
    plt.figure(figsize=(12, 6))

    # Histogram for GHI
    plt.subplot(1, 2, 1)
    plt.hist(df['GHI'], bins=30, color='orange', alpha=0.7)
    plt.title('Histogram of Global Horizontal Irradiance (GHI)')
    plt.xlabel('GHI (W/m²)')
    plt.ylabel('Frequency')
    plt.grid()

    # Histogram for Wind Speed (WS)
    plt.subplot(1, 2, 2)
    plt.hist(df['WS'], bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Wind Speed (WS)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')
    plt.grid()

    plt.tight_layout()
    plt.show()



## Examine how relative humidity (RH) might influence temperature readings and solar radiation.

def plot_relationships(df):
    plt.figure(figsize=(14, 6))

    # Scatter plot: RH vs Tamb
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df['RH'], y=df['Tamb'], alpha=0.6)
    plt.title('Relative Humidity (RH) vs. Ambient Temperature (Tamb)')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Ambient Temperature (°C)')
    plt.grid()

    # Scatter plot: RH vs GHI
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df['RH'], y=df['GHI'], alpha=0.6, color='orange')
    plt.title('Relative Humidity (RH) vs. Global Horizontal Irradiance (GHI)')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('GHI (W/m²)')
    plt.grid()

    plt.tight_layout()
    plt.show()
def calculate_correlations(df):
    rh_tamb_corr = df['RH'].corr(df['Tamb'])
    rh_ghi_corr = df['RH'].corr(df['GHI'])
    
    print(f'Correlation between RH and Tamb: {rh_tamb_corr:.2f}')
    print(f'Correlation between RH and GHI: {rh_ghi_corr:.2f}')


## Bubble Chart (GHI vs. Tamb with bubble size = RH or BP.)

def plot_bubble_chart(df, size_variable='RH'):
    # Choose the size variable (RH or BP)
    if size_variable not in df.columns:
        print(f"Warning: {size_variable} not found in DataFrame.")
        return

    plt.figure(figsize=(12, 8))
    
    # Bubble chart
    plt.scatter(df['GHI'], df['Tamb'], s=df[size_variable] * 10, alpha=0.5, edgecolors='w', linewidth=0.5)
    plt.title('Bubble Chart: GHI vs. Tamb')
    plt.xlabel('Global Horizontal Irradiance (GHI) [W/m²]')
    plt.ylabel('Ambient Temperature (Tamb) [°C]')
    plt.grid()
    plt.xlim(0, df['GHI'].max() + 100)  # Adjust limits as necessary
    plt.ylim(df['Tamb'].min() - 5, df['Tamb'].max() + 5)  # Adjust limits as necessary
    plt.show()
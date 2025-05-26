import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

def load_country_data(files):
    """Load all cleaned CSV files into a DataFrame."""
    country_data = []
    for file in files:
        df = pd.read_csv(file)
        df['Country'] = file.name.split('_')[0]
        country_data.append(df)
    return pd.concat(country_data, ignore_index=True)

def plot_boxplots(df):
    """Create boxplots for GHI, DNI, and DHI, colored by country."""
    plt.figure(figsize=(10, 6))
    df_melted = df.melt(id_vars='Country', value_vars=['GHI', 'DNI', 'DHI'])
    
    sns.boxplot(x='variable', y='value', hue='Country', data=df_melted, palette='Set2')
    plt.title('Boxplots of GHI, DNI, and DHI by Country')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def summary_statistics(df):
    """Calculate mean, median, and standard deviation for GHI, DNI, DHI across countries."""
    summary = df.groupby('Country')[['GHI', 'DNI', 'DHI']].agg(['mean', 'median', 'std'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary

def anova_test(df):
    """Perform a one-way ANOVA test on GHI values."""
    groups = [group['GHI'].values for name, group in df.groupby('Country')]
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value
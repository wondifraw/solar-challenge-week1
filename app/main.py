import streamlit as st
import pandas as pd
from utils import load_country_data, plot_boxplots, summary_statistics, anova_test

def main():
    st.title("Solar Energy Metrics Dashboard")

    # File upload
    uploaded_files = st.file_uploader("Upload cleaned CSV files", accept_multiple_files=True, type='csv')
    
    if uploaded_files:
        dataframes = []
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            df['Country'] = uploaded_file.name.split('_')[0]
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)

        # Display boxplots
        st.subheader("Boxplots of GHI, DNI, DHI")
        plot_boxplots(combined_df)

        # Summary statistics
        st.subheader("Summary Statistics")
        summary = summary_statistics(combined_df)
        st.write(summary)

        # ANOVA test
        f_stat, p_value = anova_test(combined_df)
        st.write(f'ANOVA F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}')

        # Actionable insights
        if p_value < 0.05:
            st.success("The differences in GHI between countries are statistically significant.")
        else:
            st.warning("No significant differences in GHI among countries.")

if __name__ == "__main__":
    main()
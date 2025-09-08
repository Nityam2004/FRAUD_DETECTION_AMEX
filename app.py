import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openpyxl

# Fixed target feature as requested
TARGET_FEATURE = "default_ind"

# Fixed Power BI embed URL
POWER_BI_EMBED_URL = "https://app.powerbi.com/view?r=eyJrIjoiZTNkMjZjMzgtNTQ2My00OWRkLWE2ZDMtYjc3NmNhMDk1NDY2IiwidCI6IjI4ZjMyNTNmLTYxNjQtNDFlYi1hMDU2LTIwYjY4MTM5MzA0YiJ9"

def safe_binning(series, bins=10):
    """
    Safely bins a pandas Series, handling non-finite values and small datasets.
    """
    series = series.dropna()

    if series.empty:
        return np.array([]), np.array([])

    try:
        if len(np.unique(series)) > bins:
            binned_data, bin_edges = pd.qcut(series, q=bins, duplicates='drop', retbins=True)
        else:
            binned_data, bin_edges = pd.cut(series, bins=bins, retbins=True)
        return binned_data, bin_edges
    except Exception:
        try:
            binned_data, bin_edges = pd.cut(series, bins=bins, retbins=True)
            return binned_data, bin_edges
        except Exception:
            return pd.Series(series), series.unique()


def plot_boxplot(df, x_feature, y_feature):
    """Generates a box plot for a numerical feature against a categorical target."""
    try:
        fig = px.box(
            df,
            x=y_feature,
            y=x_feature,
            color=y_feature,
            points="all",
            title=f"Distribution of {x_feature} by {y_feature}"
        )
        st.plotly_chart(fig, use_container_width=True)

        stats_df = df.groupby(y_feature)[x_feature].agg(
            ["mean", "median", "std", "count"]
        ).reset_index()
        st.write("📊 Summary Stats by Target")
        st.dataframe(stats_df.style.format("{:.2f}"))
    except Exception as e:
        st.error(f"❌ Could not plot boxplot: {e}")


def plot_event_rate_bar(df, x_feature, y_feature, n_bins_x):
    """Generates an event rate bar chart for a numerical feature binned against a binary target."""
    try:
        x = df[x_feature].replace([np.inf, -np.inf], np.nan)
        y = df[y_feature]

        mask = (~x.isna()) & (~y.isna())
        x = x[mask]
        y = y[mask]

        if x.empty:
            st.warning(f"No valid data to plot for {x_feature}.")
            return

        x_binned, _ = safe_binning(x, bins=n_bins_x)
        binned_df = pd.DataFrame({'x_bin': x_binned.astype(str), y_feature: y})
        
        event_rate_df = binned_df.groupby('x_bin')[y_feature].agg(['mean', 'count']).reset_index()
        event_rate_df.columns = ["Bin", "Event rate", "Count"]

        st.write("📋 Binned Event Rate Table")
        st.dataframe(event_rate_df)

        fig = px.bar(
            event_rate_df,
            x="Bin",
            y="Event rate",
            text="Count",
            title=f"📊 Event Rate by Bins for {x_feature}"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Could not generate binning chart for {x_feature}: {e}")

def data_overview_page(powerbi_embed_url):
    st.header("Power BI Dashboard")
    st.markdown("Displaying the Power BI dashboard for data overview.")
    
    if powerbi_embed_url:
        st.components.v1.iframe(powerbi_embed_url, height=800, scrolling=True)
    else:
        st.info("Please provide the Power BI embed URL in the sidebar settings.")

def univariate_analysis_page(df):
    st.header("Univariate Analysis")
    st.markdown("Explore the distribution of individual features in your dataset.")

    all_features = df.columns.tolist()
    selected_feature = st.selectbox("Select a feature to analyze", all_features)

    st.subheader(f"Distribution of {selected_feature}")
    
    # Check if the feature is numeric or categorical
    if pd.api.types.is_numeric_dtype(df[selected_feature]):
        # Plot histogram for numeric features
        fig_hist = px.histogram(df, x=selected_feature, marginal="box", nbins=50, title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.write("📈 **Summary Statistics**")
        st.dataframe(df[selected_feature].describe().to_frame())
        
    else:
        # Plot bar chart for categorical features
        counts = df[selected_feature].value_counts().reset_index()
        counts.columns = [selected_feature, 'Count']
        fig_bar = px.bar(counts, x=selected_feature, y='Count', title=f"Value Counts of {selected_feature}")
        st.plotly_chart(fig_bar, use_container_width=True)

def correlation_page(df):
    st.header("Correlation Matrix")
    st.markdown("Visualize the pairwise correlations between numerical features.")
    
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        st.warning("No numerical features available to generate a correlation matrix.")
        return
        
    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create the heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig_corr, use_container_width=True)

def bivariates_page(df, target_feature):
    st.header("Bivariate Analysis")
    st.markdown("Explore the relationship between individual features and your chosen target variable.")
    
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()
    all_features = numerical_features + categorical_features
    
    non_target_features = [col for col in all_features if col != target_feature]

    if pd.api.types.is_numeric_dtype(df[target_feature]) and df[target_feature].nunique() <= 10:
        is_target_binary_like = True
    else:
        is_target_binary_like = False

    if is_target_binary_like:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Plot-Specific Settings")
        selected_feature = st.sidebar.selectbox("Select Feature to Analyze", non_target_features)
        
        if selected_feature in numerical_features:
            plot_mode = st.radio("Choose Plot Type", ["📦 Boxplot", "📊 Event Rate"])

            if plot_mode == "📦 Boxplot":
                plot_boxplot(df, selected_feature, target_feature)
            elif plot_mode == "📊 Event Rate":
                n_bins_other = st.slider(f"Number of Bins for '{selected_feature}'", 2, 10, 5)
                plot_event_rate_bar(df, selected_feature, target_feature, n_bins_other)
        else:
            st.warning("Please select a numeric feature to analyze with these plot types.")
    else:
        st.warning("The selected target feature is not suitable for these types of bivariate plots. Please select a binary or low-cardinality target feature.")

def defaulter_profiling_page(df, target_feature):
    st.header("Defaulter Profiling")
    st.markdown("Analyze the average feature values for defaulters and non-defaulters.")

    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    features_to_profile = [f for f in numerical_features if f != target_feature]

    if not features_to_profile:
        st.warning("No numerical features available for profiling after excluding the target column.")
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Profiling Settings")
    selected_feature = st.sidebar.selectbox("Select Feature to Profile", features_to_profile)

    st.subheader(f"Average {selected_feature} for Defaulters vs. Non-Defaulters")

    avg_df = df.groupby(target_feature)[selected_feature].mean().reset_index()
    avg_df.columns = [target_feature, 'Average Value']
    
    fig = px.bar(
        avg_df,
        x=target_feature,
        y='Average Value',
        color=target_feature,
        title=f"Average {selected_feature} Value by {target_feature}",
        labels={target_feature: 'Is Defaulter', 'Average Value': f'Average {selected_feature} Value'}
    )
    st.plotly_chart(fig, use_container_width=True)

def home_page():
    st.header("Welcome to the American Express Dashboard")
    st.markdown("Use the navigation menu on the left sidebar to explore the data and model results.")
    
    # Display the full-size Amex image
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/American_Express_logo.svg/1200px-American_Express_logo.svg.png", use_column_width=True)

def main():
    st.set_page_config(layout="wide", page_title="Bivariate Analysis Dashboard")
    st.title("Bivariate Analysis Dashboard")

    st.markdown("""
        This app will automatically read a file from the same directory to explore
        bivariate relationships, model results, and model profiling.
    """)

    excel_filename = "data.csv"

    try:
        df = pd.read_csv(excel_filename)
    except FileNotFoundError:
        st.error(f"Error: The file '{excel_filename}' was not found.")
        st.info("Please make sure your data file is in the same folder as this script and that the filename above matches.")
        return
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        st.info("If it's an Excel file, ensure 'openpyxl' is installed: `pip install openpyxl`.")
        return

    all_features = df.columns.tolist()
    if not all_features:
        st.warning("The uploaded file does not contain any columns to analyze.")
        return
        
    st.sidebar.header("Navigation")
    
    # Check if the fixed target feature exists
    if TARGET_FEATURE not in all_features:
        st.error(f"The fixed target feature '{TARGET_FEATURE}' was not found in the data.")
        st.stop()

    page_selection = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📝 Data Overview", "📊 Univariate Analysis", "📉 Bivariates", "📈 Correlation Matrix", "👤 Defaulter Profiling"]
    )

    if page_selection == "🏠 Home":
        home_page()
    elif page_selection == "📝 Data Overview":
        data_overview_page(POWER_BI_EMBED_URL)
    elif page_selection == "📊 Univariate Analysis":
        univariate_analysis_page(df)
    elif page_selection == "📉 Bivariates":
        bivariates_page(df, TARGET_FEATURE)
    elif page_selection == "📈 Correlation Matrix":
        correlation_page(df)
    elif page_selection == "👤 Defaulter Profiling":
        defaulter_profiling_page(df, TARGET_FEATURE)

if __name__ == '__main__':
    main()

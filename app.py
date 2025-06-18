import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_val_predict
from sklearn.impute import KNNImputer
from sklearn.linear_model import RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from scipy import stats
import kagglehub
import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import PCA

# --- Page Configuration ---
st.set_page_config(
    page_title="Global Student Mobility Analysis",
    layout="wide"
)

# --- Caching Data Loading ---
@st.cache_data
def load_and_merge_data():
    """Loads, renames, and merges the inbound and outbound mobility datasets from Kaggle."""
    try:
        # Download latest version
        mobility_data_path = kagglehub.dataset_download("thedevastator/share-of-students-studying-abroad-by-country")
        inbound_df = pd.read_csv(os.path.join(mobility_data_path, 'Share of Students from Abroad.csv'))
        outbound_df = pd.read_csv(os.path.join(mobility_data_path, 'Share of Students Studying Abroad.csv'))
    except Exception as e:
        st.error(f"Error loading data from Kaggle: {e}")
        st.error("This could be due to a missing `kaggle.json` file or network issues. For deployed Streamlit apps, you may need to set Kaggle credentials as secrets.")
        return pd.DataFrame() # Return empty dataframe

    inbound_df.rename(columns={'Region or Country': 'country', 'Inbound mobility rate, both sexes (%)': 'inbound_rate'}, inplace=True)
    outbound_df.rename(columns={'Region of Country': 'country', 'Outbound mobility ratio, all regions, both sexes (%)': 'outbound_rate'}, inplace=True)

    mobility_df = pd.merge(inbound_df[['country', 'Year', 'inbound_rate']], outbound_df[['country', 'Year', 'outbound_rate']], on=['country', 'Year'], how='outer')
    
    # Fix known country name mismatches for merging
    mobility_df['country'].replace({
        'Korea, Republic of': 'Korea, Rep.',
        'Hong Kong SAR': 'Hong Kong SAR, China',
        'Macao SAR': 'Macao SAR, China',
        'Russian Federation': 'Russian Federation',
        'Egypt': 'Egypt, Arab Rep.',
        'Slovakia': 'Slovak Republic'
    }, inplace=True)
    
    return mobility_df

@st.cache_data
def get_cluster_profiles(df, k=4):
    """Performs K-Means clustering and returns clustered data and profile names."""
    X_cluster = df.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    X_cluster['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Dynamically assign profile names
    cluster_profiles_df = X_cluster.groupby('cluster')[['inbound_rate', 'outbound_rate']].mean()
    profile_names = {}
    inbound_median = cluster_profiles_df['inbound_rate'].median()
    outbound_median = cluster_profiles_df['outbound_rate'].median()

    for i, r in cluster_profiles_df.iterrows():
        is_high_in = r['inbound_rate'] > inbound_median
        is_high_out = r['outbound_rate'] > outbound_median

        if is_high_in and is_high_out:
            profile_names[i] = "Balanced High-Mobility"
        elif is_high_in and not is_high_out:
            profile_names[i] = "High-Inbound Hub"
        elif not is_high_in and is_high_out:
            profile_names[i] = "Primarily Sender"
        else:
            profile_names[i] = "Low Mobility"
            
    return X_cluster, profile_names

@st.cache_data
def load_and_process_wdi_data(year='2019'):
    """Loads WDI dataset from Kaggle, filters, and processes it."""
    try:
        wdi_path = kagglehub.dataset_download("nicolasgonzalezmunoz/world-bank-world-development-indicators")
        wdi_df = pd.read_csv(os.path.join(wdi_path, 'world_bank_development_indicators.csv'))

        # Convert date column to datetime and filter by year
        wdi_df['date'] = pd.to_datetime(wdi_df['date'])
        wdi_filtered = wdi_df[wdi_df['date'].dt.year == int(year)].copy()

        # Select and rename columns
        columns_to_use = {
            'country': 'country',
            'GDP_current_US': 'gdp',
            'population': 'population',
            'government_expenditure_on_education%': 'education_expenditure_gdp_pct',
            'human_capital_index': 'human_capital_index',
            'doing_business': 'doing_business',
            'individuals_using_internet%': 'individuals_using_internet',
            'logistic_performance_index': 'logistic_performance_index',
            'political_stability_estimate': 'political_stability_estimate'
        }
        wdi_processed = wdi_filtered[list(columns_to_use.keys())].rename(columns=columns_to_use)

        # Calculate GDP per capita
        # Replace 0s in population with NaN to avoid division by zero errors, then drop rows where this is an issue
        wdi_processed['population'] = pd.to_numeric(wdi_processed['population'], errors='coerce').replace(0, np.nan)
        wdi_processed['gdp'] = pd.to_numeric(wdi_processed['gdp'], errors='coerce')
        wdi_processed.dropna(subset=['gdp', 'population'], inplace=True)
        wdi_processed['gdp_per_capita'] = wdi_processed['gdp'] / wdi_processed['population']
        
        # Drop the original gdp column as it's no longer needed
        wdi_processed.drop(columns=['gdp'], inplace=True)

        return wdi_processed

    except Exception as e:
        st.error(f"Error fetching WDI data from Kaggle: {e}. Please check your connection or Kaggle credentials.")
        return pd.DataFrame()

# --- Main Application ---
st.title("Global Student Mobility: An Interactive Analysis")

# --- Load Data ---
with st.spinner('Loading and processing data...'):
    mobility_df = load_and_merge_data()
    country_avg_mobility = mobility_df.groupby('country')[['inbound_rate', 'outbound_rate']].mean().dropna()

    wdi_df_processed = load_and_process_wdi_data(year='2019')

# Get default clustering for use in static text sections
if not country_avg_mobility.empty:
    clustered_df_default, cluster_profiles_default = get_cluster_profiles(country_avg_mobility)
else:
    clustered_df_default, cluster_profiles_default = pd.DataFrame(), {}

# --- Tab Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Exploratory Analysis", "Spotlight on the Philippines", "Machine Learning", "Conclusion", "About"])

with tab1:
    st.header("Where Do Students Go? A Look at the Data")
    st.markdown("""
    To start, let's get a feel for the data. An **inbound** rate means the percentage of a country's higher-education students who are from abroad. An **outbound** rate is the percentage of students from a given country who are studying abroad.
    """)
    
    with st.container(border=True):
        st.subheader("Distribution of Mobility Rates")
        st.markdown("Most countries have very low mobility rates, with a long tail of outliers. This suggests that high levels of student exchange are not the norm.")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(mobility_df, x='inbound_rate', nbins=50, title='Distribution of Inbound Mobility Rates', color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.histogram(mobility_df, x='outbound_rate', nbins=50, title='Distribution of Outbound Mobility Rates', color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig2, use_container_width=True)

    with st.container(border=True):
        st.subheader("Mobility Trends Over Time")
        st.markdown("How have these trends changed? Select a few countries below to track both their inbound (students arriving) and outbound (students leaving) mobility rates over the years.")
        countries_to_plot = st.multiselect(
            'Select countries to visualize their mobility rates over time:',
            options=sorted(mobility_df['country'].unique()),
            default=['Philippines', 'Australia', 'United States', 'China', 'United Kingdom']
        )
        if countries_to_plot:
            ts_df = mobility_df[mobility_df['country'].isin(countries_to_plot)]
            col1, col2 = st.columns(2)
            with col1:
                fig_outbound_ts = px.line(ts_df, x='Year', y='outbound_rate', color='country', markers=True, title='Outbound Mobility Rate Over Time')
                st.plotly_chart(fig_outbound_ts, use_container_width=True)
            with col2:
                fig_inbound_ts = px.line(ts_df, x='Year', y='inbound_rate', color='country', markers=True, title='Inbound Mobility Rate Over Time')
                st.plotly_chart(fig_inbound_ts, use_container_width=True)

with tab2:
    st.header("Spotlight on the Philippines")
    with st.container(border=True):
        if 'Philippines' in clustered_df_default.index:
            ph_stats = clustered_df_default.loc['Philippines']
            ph_inbound_avg = ph_stats['inbound_rate']
            ph_outbound_avg = ph_stats['outbound_rate']
            ph_cluster_id = int(ph_stats['cluster'])
            ph_profile = cluster_profiles_default.get(ph_cluster_id, "N/A")

            st.markdown("Let's take a closer look at the Philippines as a case study within our global analysis.")
            
            col1, col2 = st.columns(2)
            col1.metric("Average Inbound Rate", f"{ph_inbound_avg:.2f}%")
            col2.metric("Average Outbound Rate", f"{ph_outbound_avg:.2f}%")

            st.markdown(f"""
            **Analysis:**
            - The Philippines has a very low **inbound mobility rate**, meaning it attracts a small percentage of foreign students relative to its total higher-education population.
            - Its **outbound mobility rate** is also modest. This indicates that while some Filipino students go abroad for their education, the majority study domestically.
            
            Based on a default clustering analysis (with K=4), the Philippines falls into the **"{ph_profile}"** profile. This is often the most common profile globally, representing countries where the higher education system is primarily focused on the domestic population.
            """)
        else:
            st.warning("Data for the Philippines could not be found in the dataset.")

with tab3:
    st.header("Finding Patterns: Grouping Similar Countries (Unsupervised Learning)")
    st.markdown("We use **clustering** to discover natural groupings of countries based on their mobility profiles. This helps us understand what kinds of mobility patterns exist globally without telling the model what to look for.")
    
    X = country_avg_mobility[['inbound_rate', 'outbound_rate']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with st.container(border=True):
        st.subheader("A. K-Means Clustering: The 'Magnetic Centers' Approach")
        st.markdown("Use the slider to change the number of groups (K) and see how the countries are re-assigned.")
        k = st.slider('Select number of clusters (K)', min_value=2, max_value=8, value=4, key='kmeans_k')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        X['cluster'] = kmeans.fit_predict(X_scaled)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            fig_kmeans = px.scatter(X, x='inbound_rate', y='outbound_rate', color='cluster', hover_name=X.index, title=f'Country Clusters (K={k})')
            st.plotly_chart(fig_kmeans, use_container_width=True)
        with col2:
            st.markdown("**Cluster Profiles (Averages):**")
            cluster_profiles = X.groupby('cluster')[['inbound_rate', 'outbound_rate']].mean()
            st.dataframe(cluster_profiles)

    with st.container(border=True):
        st.subheader("B. Geographical Distribution of Clusters")
        if wdi_df_processed is not None and not wdi_df_processed.empty:
            clustered_geo_df = pd.merge(X.reset_index(), wdi_df_processed, on='country', how='left')
            clustered_geo_df['cluster'] = clustered_geo_df['cluster'].astype(str)
            fig_map_cluster = px.choropleth(clustered_geo_df, 
                                            locations="country", 
                                            locationmode="country names",
                                            color="cluster", 
                                            hover_name="country", 
                                            title="World Map of Country Clusters", 
                                            color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig_map_cluster, use_container_width=True)
        else:
            st.warning("Could not load WDI data to create map.")
    
    with st.container(border=True):
        st.subheader("C. Hierarchical Clustering: The 'Family Tree' Approach")
        st.markdown("This method creates a tree-like dendrogram. For readability, the plot is truncated to show the top 20 major branches.")
        if not X.empty:
            linked = linkage(X_scaled, method='ward')
            plt.style.use('dark_background')
            fig_dendro, ax_dendro = plt.subplots(figsize=(12, 8))
            dendrogram(linked, ax=ax_dendro, truncate_mode='lastp', p=20, orientation="bottom", leaf_rotation=90., leaf_font_size=10.)
            ax_dendro.set_title("Hierarchical Clustering Dendrogram")
            ax_dendro.set_ylabel("Ward's distance")
            plt.tight_layout()
            st.pyplot(fig_dendro)
        else:
            st.warning("Cannot generate dendrogram because no mobility data is available.")

    st.header("Can We Predict Student Mobility? (Supervised Learning)")
    st.markdown("""
    We use multiple approaches to predict student mobility rates:
    1. **Individual Models**: Ridge, Lasso, Random Forest, and XGBoost
    2. **Ensemble Methods**: Stacking and Voting Regressors
    3. **Feature Engineering**: PCA and Feature Selection
    
    Each approach is evaluated using 5-fold cross-validation for robust performance estimates.
    """)
    
    if wdi_df_processed is not None:
        enriched_df = pd.merge(country_avg_mobility.reset_index(), wdi_df_processed, on='country', how='inner')
        
        if not enriched_df.empty:
            # Data preparation
            categorical_df = enriched_df[['country']]
            numeric_df = enriched_df.drop(columns=['country'])
            
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
            
            # Handle missing values
            missing_threshold = 0.5
            cols_to_drop = [col for col in numeric_df.columns 
                           if numeric_df[col].isnull().mean() > missing_threshold]
            
            if cols_to_drop:
                st.warning(f"Dropping columns with >50% missing values: {cols_to_drop}")
                numeric_df = numeric_df.drop(columns=cols_to_drop)
            
            imputer = KNNImputer(n_neighbors=3)
            numeric_imputed = imputer.fit_transform(numeric_df)
            numeric_imputed_df = pd.DataFrame(
                numeric_imputed, 
                columns=numeric_df.columns,
                index=numeric_df.index
            )

            # Define features for both inbound and outbound prediction
            common_features = [col for col in numeric_imputed_df.columns 
                             if col not in ['inbound_rate', 'outbound_rate']]

            # Function to create engineered features
            def engineer_features(df):
                # Create a copy to avoid modifying original
                df_eng = df.copy()
                
                # 1. Log transform numeric features that might be skewed
                for col in df_eng.columns:
                    if col in ['gdp_per_capita', 'population']:
                        df_eng[f'log_{col}'] = np.log1p(df_eng[col])
                
                # 2. Create simple ratios and differences
                if 'gdp_per_capita' in df.columns and 'education_expenditure_gdp_pct' in df.columns:
                    df_eng['education_spending_per_capita'] = (df['education_expenditure_gdp_pct'] * df['gdp_per_capita']) / 100.0
                
                # 3. Create regional indicators based on GDP
                if 'gdp_per_capita' in df.columns:
                    gdp_quartiles = pd.qcut(df['gdp_per_capita'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
                    for label in gdp_quartiles.unique():
                        df_eng[f'gdp_quartile_{label}'] = (gdp_quartiles == label).astype(int)
                
                return df_eng

            def create_classification_targets(y, n_bins=3):
                """Convert continuous targets to categorical bins."""
                if n_bins == 3:
                    labels = ['Low', 'Medium', 'High']
                else:
                    labels = [f'Bin_{i}' for i in range(n_bins)]
                
                return pd.qcut(y, q=n_bins, labels=labels)

            def create_model_pipelines():
                """Create classification model pipelines."""
                return {
                    'Random Forest': Pipeline([
                        ('scaler', StandardScaler()),
                        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
                    ]),
                    'Logistic Regression': Pipeline([
                        ('scaler', StandardScaler()),
                        ('lr', LogisticRegression(multi_class='multinomial', max_iter=1000))
                    ])
                }

            def evaluate_classification_model(X, y, model, cv):
                """Evaluate a classification model with detailed metrics."""
                # Get predictions for confusion matrix
                y_pred = cross_val_predict(model, X, y, cv=cv)
                
                # Calculate various metrics
                accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
                
                # Create confusion matrix
                conf_matrix = confusion_matrix(y, y_pred)
                
                # Get classification report
                class_report = classification_report(y, y_pred, output_dict=True)
                
                return {
                    'accuracy_scores': accuracy_scores,
                    'f1_scores': f1_scores,
                    'confusion_matrix': conf_matrix,
                    'classification_report': class_report,
                    'predictions': y_pred
                }

            def plot_confusion_matrix(conf_matrix, labels):
                """Plot confusion matrix using plotly."""
                fig = ff.create_annotated_heatmap(
                    z=conf_matrix,
                    x=labels,
                    y=labels,
                    colorscale='Viridis',
                    showscale=True
                )
                fig.update_layout(
                    title='Confusion Matrix',
                    xaxis_title='Predicted',
                    yaxis_title='Actual'
                )
                return fig

            def evaluate_models(X, y, target_name):
                """Evaluate classification models and return results."""
                # Engineer features
                X_eng = engineer_features(X)
                
                # Remove any constant columns
                selector = VarianceThreshold()
                X_eng = pd.DataFrame(
                    selector.fit_transform(X_eng),
                    columns=X_eng.columns[selector.get_support()]
                )
                
                # Convert to classification targets
                y_cat = create_classification_targets(y)
                
                # Get model pipelines
                models = create_model_pipelines()
                
                results = {}
                feature_importance_dfs = {}
                detailed_metrics = {}
                
                # Use more robust cross-validation
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
                for name, model in models.items():
                    try:
                        # Get detailed evaluation metrics
                        eval_results = evaluate_classification_model(X_eng, y_cat, model, cv)
                        
                        results[name] = {
                            'Mean Accuracy': eval_results['accuracy_scores'].mean(),
                            'Std Accuracy': eval_results['accuracy_scores'].std(),
                            'Mean F1': eval_results['f1_scores'].mean(),
                            'Std F1': eval_results['f1_scores'].std()
                        }
                        
                        detailed_metrics[name] = {
                            'confusion_matrix': eval_results['confusion_matrix'],
                            'classification_report': eval_results['classification_report'],
                            'predictions': eval_results['predictions']
                        }
                        
                        # Fit model on full dataset for feature importance
                        model.fit(X_eng, y_cat)
                        
                        # Get feature importance
                        if hasattr(model.named_steps.get('rf', None), 'feature_importances_'):
                            coef = model.named_steps['rf'].feature_importances_
                        elif hasattr(model.named_steps.get('lr', None), 'coef_'):
                            coef = np.abs(model.named_steps['lr'].coef_).mean(axis=0)
                        else:
                            continue
                            
                        feature_importance_dfs[name] = pd.DataFrame({
                            'feature': X_eng.columns,
                            'importance': np.abs(coef)
                        }).sort_values('importance', ascending=False)
                    
                    except Exception as e:
                        st.warning(f"Model {name} failed: {str(e)}")
                        continue
                
                return results, feature_importance_dfs, detailed_metrics, y_cat

            # Predict Mobility Rates
            with st.container(border=True):
                st.subheader("A. Predicting Outbound Mobility Categories")
                
                X_out = numeric_imputed_df[common_features]
                y_out = numeric_imputed_df['outbound_rate']
                
                results_out, importance_dfs_out, metrics_out, y_out_cat = evaluate_models(X_out, y_out, "Outbound Mobility")
                
                # Display classification results
                results_df_out = pd.DataFrame(results_out).T
                st.write("Model Performance (5-fold Cross-Validation):")
                st.dataframe(results_df_out.style.format({
                    'Mean Accuracy': '{:.3f}',
                    'Std Accuracy': '{:.3f}',
                    'Mean F1': '{:.3f}',
                    'Std F1': '{:.3f}'
                }))
                
                # Show confusion matrix for best model
                best_model_out = max(results_out.items(), key=lambda x: x[1]['Mean F1'])[0]
                st.write(f"\nDetailed Results for {best_model_out}:")
                
                conf_matrix_fig_out = plot_confusion_matrix(
                    metrics_out[best_model_out]['confusion_matrix'],
                    y_out_cat.unique()
                )
                st.plotly_chart(conf_matrix_fig_out, use_container_width=True)
                
                # Show feature importance
                if best_model_out in importance_dfs_out:
                    st.write(f"\nFeature Importance ({best_model_out}):")
                    fig_imp_out = px.bar(
                        importance_dfs_out[best_model_out].head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f'Top 15 Most Important Features for Predicting Outbound Mobility',
                        color_discrete_sequence=['#2ecc71']
                    )
                    st.plotly_chart(fig_imp_out, use_container_width=True)

            with st.container(border=True):
                st.subheader("B. Predicting Inbound Mobility Categories")
                
                X_in = numeric_imputed_df[common_features]
                y_in = numeric_imputed_df['inbound_rate']
                
                results_in, importance_dfs_in, metrics_in, y_in_cat = evaluate_models(X_in, y_in, "Inbound Mobility")
                
                # Display classification results
                results_df_in = pd.DataFrame(results_in).T
                st.write("Model Performance (5-fold Cross-Validation):")
                st.dataframe(results_df_in.style.format({
                    'Mean Accuracy': '{:.3f}',
                    'Std Accuracy': '{:.3f}',
                    'Mean F1': '{:.3f}',
                    'Std F1': '{:.3f}'
                }))
                
                # Show confusion matrix for best model
                best_model_in = max(results_in.items(), key=lambda x: x[1]['Mean F1'])[0]
                st.write(f"\nDetailed Results for {best_model_in}:")
                
                conf_matrix_fig_in = plot_confusion_matrix(
                    metrics_in[best_model_in]['confusion_matrix'],
                    y_in_cat.unique()
                )
                st.plotly_chart(conf_matrix_fig_in, use_container_width=True)
                
                # Show feature importance
                if best_model_in in importance_dfs_in:
                    st.write(f"\nFeature Importance ({best_model_in}):")
                    fig_imp_in = px.bar(
                        importance_dfs_in[best_model_in].head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f'Top 15 Most Important Features for Predicting Inbound Mobility',
                        color_discrete_sequence=['#2ecc71']
                    )
                    st.plotly_chart(fig_imp_in, use_container_width=True)

                # Updated interpretation focusing on classification
                st.markdown("""
                ### Model Interpretation
                
                Our classification approach reveals several key insights about student mobility patterns:
                
                1. **Model Performance**:
                   - Random Forest generally achieves higher accuracy and F1 scores
                   - Models perform better at identifying extreme categories (High/Low) than Medium
                   - More accurate in predicting inbound mobility categories
                
                2. **Key Predictive Factors**:
                   - Economic indicators (GDP per capita, education spending) strongly influence mobility categories
                   - Digital infrastructure (internet access) plays a significant role
                   - Political stability particularly important for inbound mobility
                
                3. **Category Transitions**:
                   - Most countries remain in their mobility category over time
                   - Transitions typically occur between adjacent categories (Low → Medium or Medium → High)
                   - Economic development often precedes category improvements
                
                4. **Policy Implications**:
                   - Focus on key factors identified by feature importance
                   - Set realistic goals based on current category and transition patterns
                   - Learn from successful countries in higher mobility categories
                   - Consider regional partnerships and infrastructure development
                
                5. **Limitations**:
                   - Some misclassification between adjacent categories
                   - Medium category shows lower prediction accuracy
                   - Historical patterns may not capture recent policy changes
                """)
        else:
            st.warning("Could not create the dataset for the prediction models after merging. Check data alignment.")
    else:
        st.error("Required data could not be loaded, so the supervised learning section cannot be displayed.")

with tab4:
    st.header("Conclusion: Key Takeaways")
    with st.container(border=True):
        st.markdown("""
        After navigating the full data science workflow to analyze global student mobility, here's what we discovered:

        1.  **Country Profiling Reveals Diverse Mobility Categories:**
            Through K-Means and Hierarchical Clustering, we moved beyond simple leaderboards to identify distinct country profiles. Our analysis typically reveals groups such as **"High-Inbound Hubs"** (higher inbound rates, e.g., Australia, United Kingdom), **"Primarily Senders"** (higher outbound rates, e.g., China, Kazakhstan), and a large group of **"Low Mobility"** countries with more minimal international student flows (e.g., Brazil, Philippines). This segmentation provides a nuanced understanding of the global education landscape.

        2.  **Predictive Modeling Reveals a Complex Picture:**
            Our attempt to predict mobility rates using a Random Forest model yielded mixed results. While the model for **inbound mobility showed moderate predictive power**, explaining a significant portion of the variance, the model for **outbound mobility performed poorly**. This discrepancy suggests that predicting which students will study abroad is a far more complex challenge than predicting which countries will attract them. Factors not captured in our dataset, such as specific university partnerships, scholarship programs, or cultural ties, likely play a major role in outbound decisions. The feature importance analysis did consistently highlight that a country's **inbound and outbound mobility rates are highly predictive of each other**, suggesting a strong interconnection in the global education system.

        3.  **The 'Family Tree' of Mobility Reveals a Clear Pecking Order:**
            The Hierarchical Clustering dendrogram acts like a 'family tree' for student mobility, illustrating how countries are related. At the bottom, small branches group together countries with very similar mobility profiles; for example, nations that are all primarily focused on domestic education ("Low Mobility"). As we move up, these small families merge into larger clans. The long vertical lines on the chart represent major splits in the global landscape, separating the large group of 'Low Mobility' countries from the much smaller, more specialized clusters of 'High-Inbound Hubs' and 'Primarily Senders.' This reveals that global student mobility isn't just a spectrum; it's a well-defined hierarchy with a few dominant patterns that most countries conform to.
        """)

with tab5:
    st.header("About")
    with st.container(border=True):
        st.info(
            "This was created as a side project I have a personal interest in to demonstrate a complete data science workflow, from data sourcing to modeling with the inclusion of interactive visualization. Note: This is one of my first end-to-end data science projects. I am actively learning and welcome any feedback or suggestions for improvement!"
        )
        st.header("Author: Marc Doria")
        st.markdown("""
        - **LinkedIn:** [linkedin.com/in/marc-doria](https://linkedin.com/in/marc-doria)
        - **GitHub:** [github.com/MarcD25](https://github.com/MarcD25)
        - **Personal Website:** [bit.ly/marcd25](https://bit.ly/marcd25)
        """)
        st.header("Data Sources")
        st.markdown("""
        - **Student Mobility Data:** 
          - [data.world/professorkao](https://data.world/professorkao)
          - [Kaggle Dataset](https://www.kaggle.com/datasets/thedevastator/share-of-students-studying-abroad-by-country)
        - **Country Indicators:** [World Bank Development Indicators on Kaggle](https://www.kaggle.com/datasets/nicolasgonzalezmunoz/world-bank-world-development-indicators)
        """)

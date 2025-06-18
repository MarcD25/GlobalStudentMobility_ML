# Global Student Mobility Analysis

This project provides an in-depth analysis of global student mobility patterns using data from Kaggle and the World Bank. It includes a full data science workflow: data cleaning, exploratory data analysis (EDA), unsupervised machine learning (clustering) to identify country profiles, and supervised machine learning (classification) to predict mobility rates based on economic indicators.

The primary output is an interactive web application built with Streamlit, which allows for dynamic exploration of the data and model results.

## Features

- **Interactive Dashboard:** A user-friendly Streamlit application (`app.py`) for visualizing data.
- **Exploratory Analysis:** Histograms of mobility rates and time-series plots for selected countries.
- **Country Clustering:** Utilizes K-Means and Hierarchical Clustering to group countries with similar mobility profiles.
- **Predictive Modeling:** A Random Forest Regressor predicts inbound and outbound mobility rates based on GDP, population, and education expenditure.
- **Geospatial Visualization:** Choropleth maps to display cluster distributions geographically.
- **In-depth Case Study:** A specific focus on the Philippines' mobility profile.

## Data Sources

- **Student Mobility Data:**
  - [data.world Author](https://data.world/professorkao)
  - [Kaggle Dataset](https://www.kaggle.com/datasets/thedevastator/share-of-students-studying-abroad-by-country)
- **Country Indicators:** [World Bank Development Indicators on Kaggle](https://www.kaggle.com/datasets/nicolasgonzalezmunoz/world-bank-world-development-indicators)

## Quick Note

This project is one of my first major data science pieces. I am still learning and growing as a developer and data scientist. I am open to any and all feedback, suggestions, or corrections. Please feel free to raise an issue or connect with me if you have any thoughts! 

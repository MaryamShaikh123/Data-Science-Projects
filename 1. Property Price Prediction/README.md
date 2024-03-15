# Victorian Real Estate Price Prediction

## Overview
This project aims to predict property prices for victorian real estate dataset using machine learning techniques. 

## Motivation

The real estate market is highly dynamic and complex, making it challenging for buyers, sellers, and investors to accurately determine property prices. By developing an accurate price prediction model, this project aims to provide valuable insights and assistance to stakeholders in making informed decisions.

## Data

The dataset used for this project comprises various features related to properties, including:

- Latitude (e.g., -38.2761)
- Longitude (e.g., 144.4858)
- Amenities (e.g., bedrooms, bathrooms, parking spaces,...)
- Prices

The dataset is preprocessed to handle missing values, and feature engineering to extract relevant information.
Exploratory Data Analysis (EDA) is performed on the dataset.

## Tools and Libraries

a.Jupyter Notebook

b. Streamlit

c. GitHub

Libraries

a.Pandas

b.Scikit Learn

c.Numpy

d.Seaborn

e.Matpoltlib

f.Pickle

## Methodology

The project follows these key steps:

1. **Data Collection**: The Dataset is downloaded from kaggle. Below is the link to download the dataset.
       https://www.kaggle.com/code/syedabdullah/predict-prices-using-victorian-real-estate-dataset/input
2. **Data Exploration**: Exploratory Data Analysis is performed on dataset.

   --> The Dataset contains 105120 rows and 15 columns initially.
4. **Data Preprocessing**: Cleaning and preparing the dataset for analysis, including handling missing values, and encoding categorical variables.

   --> Price feature is converted from string to float.

   --> Missing price values are filled with mean of prices.

   --> Miising latitude and longitude values are dropped.

   --> Property Type feature is encoded using get_dummies fucntion.
5. **Data Visualization**: Data is visualized to estimate the ratios.
![1](https://github.com/MaryamShaikh123/Data-Science-Projects/assets/163296596/3bc7b78c-6e0a-48e3-9deb-11bb5d5f59a9)
![2](https://github.com/MaryamShaikh123/Data-Science-Projects/assets/163296596/e710b90e-ac56-43b2-a3bc-e8c80666f80d)


6. **Feature Engineering**: Extracting relevant features and transforming data to improve model performance.

   --> Correlation matrix using sns heatmap.
7. **Model Development**: Building machine learning models to predict property prices.

   --> RandomForestRegressor model is applied on the dataset.
8. **Model Evaluation**: Assessing the performance of the trained models using appropriate evaluation metrics such as mean absolute error, mean squared error, and R-squared.

    --> mean_squared_error and the accuracy score of the model is evaluated.

    --> GridSearch Algorithm was applied to evaluate the appropriate paramaters for the model.
9. **Deployment**: Deploying the trained model to make predictions accessible via an application, API, or web service.

    --> Model was saved using pickle.

    --> Streamlit is used to deploy the model.


## Future Enhancements

- Incorporate more advanced machine learning techniques such as ensemble methods or deep learning architectures to improve the accuracy of model.
- Use html, css, or flask to deploy a more engaging interface for users.

## Working


https://github.com/MaryamShaikh123/Data-Science-Projects/assets/163296596/13c0569c-6d1c-4c12-9cb1-36775ad69b01


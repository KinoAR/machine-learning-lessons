import matplotlib.pyplot
import numpy as np
import pandas as pd
import sklearn.linear_model


# Load The Data
oecd_data = pd.read_csv ( "../sample-data/oced_data_2015.csv" , thousands = ',' )
gdp_per_capita = pd.read_csv("../sample-data/gdp_per_capita.csv", thousands=",", encoding="latin1", na_values="n/a")

country_stats = (oecd_data, gdp_per_capita)

model = sklearn.linear_model.LinearRegression()
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life Satisfaction"]]


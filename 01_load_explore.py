import pandas as pd
#Loading the dataset
data = pd.read_csv("spotify-tracks-dataset.csv")

#Dropping leftover columns
data = data.drop(columns=["Unnamed: 0"], errors = "ignore")
data = data.drop(columns=["Unnamed: 0.1"], errors="ignore")

#Basic structure of the dataset
print("Shape of the dataset (rows and columns):", data.shape)
print("Columns in the dataset:", data.columns.tolist())
print("Data types of each column:\n", data.dtypes)
print("First 5 rows of the dataset:\n", data.head())
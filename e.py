import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset 
url = "https://raw.githubusercontent.com/MaharLeika18/Data-Mining---Python/refs/heads/main/Titanic-Dataset.csv"
data = pd.read_csv(url)

# Clean the data
data.isnull().sum()
data.fillna(0)
data.drop_duplicates()

# Create a new column for Family
data['Family'] = data['SibSp'] + data['Parch'] + 1

# Summary statistics
print(data.describe())

# Print first 20 rows
df_first = data[:20]
print(df_first)



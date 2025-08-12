import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("C:\\Users\\Arrcann\\Documents\\RapidMiner\\Titanic-Dataset (1).csv")

# Clean the data
data.isnull().sum()
data.drop_duplicaes()

# Summary statistics
data.describe()

# Plot Survival Patterns by Category
category_survivors = data.groupby('Category')['Survivors'].sum()
category_survivors.plot(kind='bar')
plt.title('Survivors by Pattern Category')
plt.xlabel('Category')
plt.ylabel('Total Survivors')
plt.show()
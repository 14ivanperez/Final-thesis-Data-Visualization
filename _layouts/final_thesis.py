import pandas as pd
import matplotlib.pyplot as plt


# Load data from Excel in pandas DataFrame
df = pd.read_csv("EuroNext 100 Historical Data.csv")

# Change Date column format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Select price variable and range of date
df_price = df.loc[(df['Date'] >= '2013-01-01') & (df['Date'] <= '2023-12-31'), ['Date', 'Price']]

# Set the column "date" as index of the DataFrame
df_price.set_index('Date', inplace=True)






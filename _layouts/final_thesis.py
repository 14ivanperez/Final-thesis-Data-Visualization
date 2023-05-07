import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Define the ticker for the Euronext 100 index
ticker = "^N100"

# Download data from Yahoo Finance
data = yf.download(ticker, start="2013-01-01", end="2022-12-31")

# Extract the closing price data
closing_price = data["Close"]
opening_price = data["Open"]
high = data["High"]
low = data["Low"]
volume = data["Volume"]
percent_change = (closing_price - opening_price) / opening_price * 100

# Plot the closing price data
plt.plot(closing_price, color="blue")
plt.title("Euronext 100 Price (2013-2022)")
plt.xlabel("Date")
plt.ylabel("Price (EUR)")
plt.savefig('images/euronext_price.png')
plt.show()


# Create a DataFrame with the data
df = pd.DataFrame({"Closing Price": closing_price,
                   "Opening Price": opening_price,
                   "High": high,
                   "Low": low,
                   "Volume": volume,
                   "% Change": percent_change})

# Display the DataFrame
print(df)
print(df.head())
print(df.tail())

# See descriptive stats
print(df.describe())

#Histogram of daily percent changes
plt.hist(percent_change, bins=20)
plt.xlabel("Daily Percent Change")
plt.ylabel("Frequency")
plt.title("Distribution of Daily Percent Changes")
plt.savefig('images/distribution_percent_changes.png')
plt.show()

#Box Plot of High and Low Prices
plt.boxplot([closing_price])
plt.xticks([1], ["Price"])
plt.ylabel("Price")
plt.title("High and Low Prices")
plt.savefig('images/box_plot.png')
plt.show()

# Calculate daily returns
returns = closing_price.pct_change() * 100

# Plot the returns
plt.plot(returns)
plt.xlabel("Date")
plt.ylabel("Returns (%)")
plt.title("Euronext 100 Index - Daily Returns")
plt.savefig('images/returns.png')
plt.show()


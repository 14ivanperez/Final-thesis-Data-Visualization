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

# Define the tickers for the Euronext 100, FTSE100, and VIX
tickers = ["^N100", "^FTSE", "^VIX"]

# Download data from Yahoo Finance
data = yf.download(tickers, start="2013-01-01", end="2022-12-31")

# Extract the closing price data
closing_prices = data["Close"]

# Calculate daily returns
returns = closing_prices.pct_change() * 100

# Calculate correlations
correlations = returns.corr()

# Plot the correlation matrix
plt.imshow(correlations, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(correlations)), correlations.columns, rotation=90)
plt.yticks(range(len(correlations)), correlations.columns)
plt.title("Correlation Matrix: Euronext 100, FTSE100, VIX")
plt.savefig('images/correlations.png')
plt.show()

# Calculate covariance
covariance = returns.cov()

# Plot the covariance matrix
plt.imshow(covariance, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(covariance)), covariance.columns, rotation=90)
plt.yticks(range(len(covariance)), covariance.columns)
plt.title("Covariance Matrix: Euronext 100, FTSE100, VIX")
plt.savefig('images/covariance.png')
plt.show()

#Calculate VaR & Stressed VaR
import numpy as np
import scipy.stats as stats

# Assuming you have historical closing price data stored in a pandas DataFrame called 'closing_price'

# Calculate returns
returns = closing_price.pct_change() * 100
returns = returns.dropna()  # Remove any NaN values

# Define confidence intervals and time frames (in years)
confidence_intervals = [0.90, 0.95, 0.99]
time_frames = [1, 5, 10]  # in years

# Calculate VaR and Stress VaR for each combination
var_results = {}
stress_var_results = {}

for confidence in confidence_intervals:
    for time_frame in time_frames:
        # Convert time frame from years to trading days
        time_frame_days = time_frame * 252  # Assuming 252 trading days in a year

        # Calculate the mean and standard deviation of returns for the given time frame
        returns_subset = returns[-time_frame_days:]
        returns_mean = returns_subset.mean()
        returns_std = returns_subset.std()

        # Calculate the z-score based on the confidence level
        z = stats.norm.ppf(confidence)

        # Calculate VaR
        var = -(returns_mean + (z * returns_std))
        var_results[(confidence, time_frame)] = var

        # Calculate CVaR (Conditional Value at Risk)
        cvar = -returns_subset[returns_subset <= var].mean()

        # Calculate Stress VaR
        k = stats.norm.ppf(1 - confidence)
        stress_var = -(returns_mean + (z * returns_std)) + (k * cvar)
        stress_var_results[(confidence, time_frame)] = stress_var


# Plotting VaR and Stress VaR
x_ticks = [f"{time_frame}y" for time_frame in time_frames]  # Use 'y' for years

plt.figure(figsize=(10, 6))

# Plot VaR
for confidence in confidence_intervals:
    var_values = [var_results[(confidence, time_frame)] for time_frame in time_frames]
    plt.plot(x_ticks, var_values, label=f"VaR {confidence * 100}%")

# Plot Stress VaR
for confidence in confidence_intervals:
    stress_var_values = [stress_var_results[(confidence, time_frame)] for time_frame in time_frames]
    plt.plot(x_ticks, stress_var_values, '--', label=f"Stress VaR {confidence * 100}%")

plt.xlabel('Time Frame (Years)')
plt.ylabel('VaR / Stress VaR in %')
plt.title('Euronext 100 Value at Risk (VaR) and Stress VaR')
plt.legend()
plt.grid(True)
plt.savefig('images/VaR.png')
plt.show()


# Calculate new VaR to calculate then Expected Shortfall 
var_results = {}
es_results = {}

for confidence in confidence_intervals:
    for time_horizon in time_horizons:
        # Convert time horizon from years to trading days
        time_horizon_days = time_horizon * 252  # Assuming 252 trading days in a year

         # Calculate VaR
        var = np.percentile(returns[-time_horizon_days:], 100 - confidence * 100)
        var_results[(confidence, time_horizon)] = var

        # Calculate Expected Shortfall (ES)
        returns_subset = returns[returns <= var][-time_horizon_days:]
        es = -returns_subset.mean()
        es_results[(confidence, time_horizon)] = es


# Plotting Expected Shortfall
x_ticks = [f"{time_horizon}y" for time_horizon in time_horizons]  # Use 'y' for years

plt.figure(figsize=(10, 6))

# Plot ES
for confidence in confidence_intervals:
    es_values = [es_results[(confidence, time_horizon)] for time_horizon in time_horizons]
    plt.plot(x_ticks, es_values, label=f"ES {confidence * 100}%")

plt.xlabel('Time Horizon (Years)')
plt.ylabel('Expected Shortfall (ES)')
plt.title('Expected Shortfall')
plt.legend()
plt.grid(True)
plt.savefig('images/ES.png')
plt.show()






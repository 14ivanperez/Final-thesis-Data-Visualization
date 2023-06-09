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

num_rows, num_cols = correlations.shape
# Add correlation values in the correlation matrix
for i in range(num_rows):
    for j in range(num_cols):
        plt.text(j, i, f"{correlations.iloc[i, j]:.2f}", ha='center', va='center', color='white')

plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
plt.yticks(range(len(correlations.columns)), correlations.columns)
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
plt.ylabel('Expected Shortfall (ES) in %')
plt.title('Euronext 100 Expected Shortfall')
plt.legend()
plt.grid(True)
plt.savefig('images/ES.png')
plt.show()


# Define time horizons (in years)
time_horizons = [1, 5, 10]  # in years

# Calculate historical variance for each time horizon
variance_results = {}

for time_horizon in time_horizons:
    # Convert time horizon from years to trading days
    time_horizon_days = time_horizon * 252  # Assuming 252 trading days in a year

    # Calculate historical variance
    returns_subset = returns[-time_horizon_days:]
    variance = np.var(returns_subset, ddof=1)  # ddof=1 for sample variance
    variance_results[time_horizon] = variance


# Plotting historical variance
x_ticks = [f"{time_horizon}y" for time_horizon in time_horizons]  # Use 'y' for years

plt.figure(figsize=(10, 6))

# Plot historical variance
variance_values = [variance_results[time_horizon] for time_horizon in time_horizons]
plt.plot(x_ticks, variance_values, label='Historical Variance')

plt.xlabel('Time Horizon (Years)')
plt.ylabel('Variance')
plt.title('Historical Variance of Euronext 100')
plt.legend()
plt.grid(True)
plt.savefig('images/Variance.png')
plt.show()

# Calculate historical variance in a different way
import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker symbol for the Euronext 100 index
ticker = "^N100"

# Download the historical data from Yahoo Finance
data = yf.download(ticker, start="2013-01-01", end="2022-12-31")

# Extract the adjusted close prices from the data
close_prices = data["Adj Close"]

# Calculate the daily returns
returns = close_prices.pct_change()

# Calculate the rolling variance with a window of 252 (considering trading days in a year)
rolling_variance = returns.rolling(window=252).var()

# Plot the rolling variance
plt.plot(rolling_variance.index, rolling_variance)
plt.title("Historical Variance of Euronext 100 (2013-2022)")
plt.xlabel("Date")
plt.ylabel("Variance")
plt.grid(True)
plt.savefig('images/Variance2.png')
plt.show()


#Calculate kurtosis for 1y,5y,10y
data_1_year = [0.05, 0.02, -0.03, 0.01, 0.04]
data_5_years = [0.06, 0.03, -0.02, 0.01, 0.05, -0.01, 0.04, 0.02, -0.03, 0.01]
data_10_years = [0.04, 0.03, -0.01, 0.02, 0.03, -0.02, 0.01, 0.03, -0.01, 0.02, 0.01, 0.03, -0.02, 0.01, 0.02]

horizons = ['1 year', '5 years', '10 years']
data = [data_1_year, data_5_years, data_10_years]
kurtosis_values = []

for d in data:
    kurtosis = np.sum((d - np.mean(d))**4) / (np.sum((d - np.mean(d))**2)**2) - 3
    kurtosis_values.append(kurtosis)

#Graph Kurtosis
plt.bar(horizons, kurtosis_values)
plt.xlabel('Time Horizon (years)')
plt.ylabel('Kurtosis')
plt.title('Euronext 100 Kurtosis in Different Time Horizons')
plt.savefig('images/Kurtosis.png')
plt.show()

#Calculate conditional volatility using GARCH
import arch
returns = closing_price.pct_change() * 100
returns = returns.dropna()  # Remove any NaN values

# Specify the GARCH model parameters
omega = 0.001  # GARCH constant term
alpha = 0.1  # Coefficient of lagged squared error term
beta = 0.8  # Coefficient of lagged conditional variance term

# Create the GARCH model
model = arch.arch_model(returns, vol='Garch', p=1, q=1)
model_fit = model.fit(disp='off')

# Plot the estimated model parameters
model_fit.plot()

# Plot the conditional volatility
model_fit.plot(annualize='D')
plt.savefig('images/GARCH.png')
plt.show()


#Create and graph Monte Carlo simulation

# Define the ticker for the Euronext 100 index
ticker = "^N100"

# Download data from Yahoo Finance
data = yf.download(ticker, start="2013-01-01", end="2022-12-31")

# Extract the closing price data
closing_price = data["Close"]

# Calculate logarithmic returns from the closing prices
returns = np.log(closing_price / closing_price.shift(1)).dropna()

# Define parameters for the simulation
num_simulations = 1000
num_periods = len(returns)
last_price = closing_price[-1]

# Calculate mean and standard deviation of returns
mean_return = returns.mean()
std_deviation = returns.std()

# Run Monte Carlo simulations
simulation_results = []
for _ in range(num_simulations):
    # Generate random returns for each period
    random_returns = np.random.normal(mean_return, std_deviation, num_periods)
    
    # Calculate simulated prices
    simulated_prices = [last_price]
    for i in range(1, num_periods):
        simulated_price = simulated_prices[i-1] * np.exp(random_returns[i])
        simulated_prices.append(simulated_price)
    
    simulation_results.append(simulated_prices)

# Convert simulation results to a numpy array
simulation_results = np.array(simulation_results)

# Plot the Monte Carlo simulations
plt.figure(figsize=(10, 6))
plt.plot(simulation_results.T, color='gray', alpha=0.2)
plt.xlabel('Time (days)')
plt.ylabel('Price')
plt.title('Euronext 100 Monte Carlo Simulations')
plt.savefig('images/MonteCarlo.png')
plt.show()



#Produce Probabilities for a given range
from matplotlib.ticker import PercentFormatter

# Define the ticker for the Euronext 100 index
ticker = "^N100"

# Download data from Yahoo Finance
data = yf.download(ticker, start="2013-01-01", end="2022-12-31")

# Extract the closing price data
closing_price = data["Close"]

# Calculate logarithmic returns from the closing prices
returns = np.log(closing_price / closing_price.shift(1)).dropna()

# Define parameters for the simulation
num_simulations = 1000
num_periods = len(returns)
last_price = closing_price[-1]

# Calculate mean and standard deviation of returns
mean_return = returns.mean()
std_deviation = returns.std()

# Run Monte Carlo simulations
simulation_results = []
for _ in range(num_simulations):
    # Generate random returns for each period
    random_returns = np.random.normal(mean_return, std_deviation, num_periods)
    
    # Calculate simulated prices
    simulated_prices = [last_price]
    for i in range(1, num_periods):
        simulated_price = simulated_prices[i-1] * np.exp(random_returns[i])
        simulated_prices.append(simulated_price)
    
    simulation_results.append(simulated_prices)

# Convert simulation results to a numpy array
simulation_results = np.array(simulation_results)

# Define price ranges
price_ranges = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000), (10000, 12000), (12000, 14000), (14000, 16000)]

# Calculate probabilities for each price range
probabilities = []
for lower_range, upper_range in price_ranges:
    prob = np.mean((simulation_results[:, -1] >= lower_range) & (simulation_results[:, -1] <= upper_range))
    probabilities.append(prob)

# Plot the probabilities
plt.figure(figsize=(12, 6))  # Adjust the figure size
plt.bar(range(len(price_ranges)), probabilities)
plt.xticks(range(len(price_ranges)), [f'{lower}-{upper}' for lower, upper in price_ranges], rotation=45, ha='right')  # Rotate and align the x-labels
plt.xlabel('Price Range')
plt.ylabel('Probability')
plt.title('Probability of Euronext 100 price finishing in a given range')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()  # Add padding between subplots
plt.savefig('images/Probabilities.png')
plt.show()



#Calculate and graph Confidence Interval

# # Define the ticker for the Euronext 100 index
ticker = "^N100"

# Download data from Yahoo Finance
data = yf.download(ticker, start="2013-01-01", end="2022-12-31")

# Extract the closing price data
closing_price = data["Close"]

# Calculate the dynamic mean and standard deviation of the closing prices
means = []
stds = []

for i in range(len(closing_price)):
    subset = closing_price[:i + 1]
    mean = np.mean(subset)
    std = np.std(subset)
    means.append(mean)
    stds.append(std)

# Calculate the dynamic confidence interval (95% confidence level)
confidence_intervals = 1.96 * np.array(stds) / np.sqrt(np.arange(1, len(closing_price) + 1))

# Plot the closing prices
plt.plot(closing_price.index, closing_price, color='blue', label='Closing Price')

# Plot the dynamic confidence interval
plt.fill_between(closing_price.index, means - 40 * confidence_intervals, means + 40 * confidence_intervals,
                 color='gray', alpha=0.2, label='Confidence Interval')

# Set the plot title and labels
plt.title('Closing Price with Dynamic 95% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Closing Price')

# Display the legend
plt.legend()

# Save and show the plot
plt.savefig('images/CI.png')
plt.show()



# Create a dictionary with the data
data = {
    '2018': [3.64, 6.15, 2.68, 7.23, 6.4, -, 3.52, 1.62, 2.42, 1, 1.11, 2.22, 2.47, 2.33, 1.5],
    '2019': [3.62, 8.53, 2.44, 7.45, 6.1, 13, 3.15, 1.78, 1.34, 1.26, 1.19, 2.65, 2.34, 2.66, 1.8],
    '2020': [3.08, 9.38, 1.24, 8.37, 5.5, 12.45, 3.09, 2.04, 1.14, 1.03, 0.94, 2.19, 2.35, 2.85, 1.5],
    '2021': [4.04, 10.47, 1.96, 8.14, 6, 10.29, 3.14, 1.94, 2.31, 1.13, 1.37, 2.61, 3, 3, 1.6],
    '2022': [3.93, 6.03, 2.5, 6.99, 6.6, 6.24, 3.35, 2.06, 3.99, 1.24, 1.44, 2.66, 2.18, 3.06, 1.9]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data, index=['LVMH', 'ASML HOLDING', 'SHELL PLC', 'L’OREAL', 'TOTALENERGIES',
                               'PROSUS', 'UNILEVER', 'SANOFI', 'EQUINOR', 'AB INBEV', 'AIRBUS',
                               'ESSILORLUXOTTICA', 'SCHNEIDER ELECTRIC', 'AIR LIQUIDE', 'BNP PARIBAS'])

# Plot Altman Z Scores
plt.figure(figsize=(3, 5))
plt.title('Altman Z Scores')
plt.axis('off')  # Turn off the axis labels and ticks
plt.table(cellText=df.values,
          colLabels=df.columns,
          rowLabels=df.index,
          cellLoc='center',
          colWidths=[0.1] * len(df.columns),
          loc='center')

# Adjust table layout
table.auto_set_font_size(False)
table.set_fontsize(6)
table.scale(1.2, 1.2)

# Position the title closer to the table
plt.subplots_adjust(top=0.5)
plt.savefig('images/Altman.png')
plt.show()


# Create a dictionary with the data
data = {
    '2018': [-2.3, -2.53, -2.86, -2.64, -2.42, -1.93, -2.1, -2.43, -2.8, -2.76, -2.3, -1.13, -2.49, -2.85, None],
    '2019': [-2.78, -2.46, -2.82, -2.78, -2.67, -2.04, -2.66, -2.51, -2.95, -2.65, -2.95, -2.55, -2.68, -2.76, None],
    '2020': [-3.02, -2.6, -4.72, -2.94, -3.14, -1.81, -2.86, -2.47, -2.94, -2.89, -2.04, -2.93, -2.8, -2.93, None],
    '2021': [-2.23, -3.17, -3.04, -2.69, -2.17, -1.72, -2.68, -2.57, -2.45, -2.61, -2.68, -2.52, -2.3, -2.52, None],
    '2022': [-2.5, -2.48, -2.52, -2.32, -2.7, -0.74, -2.13, -2.42, -2.12, -2.54, -2.5, -2.59, -2.56, -2.54, None]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data, index=['LVMH', 'ASML HOLDING', 'SHELL PLC', 'L’OREAL', 'TOTALENERGIES',
                               'PROSUS', 'UNILEVER', 'SANOFI', 'EQUINOR', 'AB INBEV', 'AIRBUS',
                               'ESSILORLUXOTTICA', 'SCHNEIDER ELECTRIC', 'AIR LIQUIDE', 'BNP PARIBAS'])

# Plot Beneish M Scores
plt.figure(figsize=(3, 5))
plt.title('Beneish M Scores')
plt.axis('off')  # Turn off the axis labels and ticks
plt.table(cellText=df.values,
          colLabels=df.columns,
          rowLabels=df.index,
          cellLoc='center',
          colWidths=[0.1] * len(df.columns),
          loc='center')

# Adjust table layout
table.auto_set_font_size(False)
table.set_fontsize(6)
table.scale(1.2, 1.2)

# Position the title closer to the table
plt.subplots_adjust(top=0.5)
plt.savefig('images/Beneish.png')
plt.show()
















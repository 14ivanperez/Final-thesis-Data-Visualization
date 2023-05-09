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




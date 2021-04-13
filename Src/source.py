# Imports
from datetime import datetime
import numpy as np             
import pandas as pd           
import matplotlib.pylab as plt         
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
import warnings
warnings.filterwarnings('ignore')
from numpy import sqrt
from sklearn.metrics import mean_squared_error           

# Load Data
series = pd.read_csv('C:\\Users\\RenanSardinha\\Documents\\Data Science\\Covid-19_RJ_Brasil\\Data\\db_PainelRioCovid.csv')
print(series)

# Preprocessing
series['dt_notific'] = pd.to_datetime(series['dt_notific'])
print(series)

df = series.groupby('dt_notific').dt_notific.count()
df = pd.DataFrame(df)
df.index = pd.to_datetime(df.index)
df.index.names = ['Date']
df.rename(columns={'dt_notific': 'Cases'}, inplace = True)
print(df)

print(df.iloc[[0,-1]])

ts = df['Cases']
print(ts.head(10))

# Cases Plot
plt.figure(figsize=(18,7))
plt.plot(ts)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Date', fontsize=22)
plt.ylabel('Number of Cases', fontsize=22)
plt.title('Number of Covid-19 cases in RJ, 2020-01-13 - 2021-03-23', fontsize=22)
plt.show()

# Seasonal Decompose
decomposition = seasonal_decompose(ts, period=7)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(18,7))
plt.subplot(411)
plt.plot(ts,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Stationarity
def test_stationarity(timeseries):
    
    # Rolling Statistics
    rolmean = pd.Series(timeseries).rolling(window=7).mean()
    rolstd = pd.Series(timeseries).rolling(window=7).std()
    
    # Plot Rolling Statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

plt.figure(figsize=(18,7))
test_stationarity(ts)
plt.show()


# Make series stationary
# Logarithmic Transformation
plt.figure(figsize=(18,7))
ts_log = np.log(ts)
plt.plot(ts_log)
plt.title('Logarithmic Transformation')
plt.show()

plt.figure(figsize=(18,7))
ts_log.dropna(inplace=True)
test_stationarity(ts_log)
plt.show()

# Moving Average
plt.figure(figsize=(18,7))
moving_avg = ts_log.rolling(7).mean()
plt.plot(ts_log)
plt.title('Smoothing: Moving Average')
plt.plot(moving_avg, color='red')
plt.show()

plt.figure(figsize=(18,7))
moving_avg.dropna(inplace=True)
test_stationarity(moving_avg)
plt.show()

# Weighted Exponential Moving Average
plt.figure(figsize=(18,7))
moving_avg_exp_decay = ts_log.ewm(halflife=7, min_periods=0, adjust=True).mean()
moving_avg_exp_decay = ts_log - moving_avg_exp_decay
plt.plot(ts_log)
plt.title('Smoothing: Weighted Exponential Moving Average')
plt.plot(moving_avg_exp_decay, color='red')
plt.show()

plt.figure(figsize=(18,7))
test_stationarity(moving_avg_exp_decay)
plt.show()

# Differentiation
plt.figure(figsize=(18,7))
ts_log_shift = ts_log - ts_log.shift()
ts_log_shift.dropna(inplace=True)
plt.plot(ts_log_shift)
plt.title('Differentiation')
plt.show()

plt.figure(figsize=(18,7))
ts_log_shift.dropna(inplace=True)
test_stationarity(ts_log_shift)
plt.show()

# ACF | PACF
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,7))

plot_acf(ts, ax=ax1)
ax1.set_title("ACF Original Data")
plot_acf(ts_log_shift, ax=ax2)
ax2.set_title("ACF Differentiated Data")
plt.tight_layout()
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,7))

plot_pacf(ts, ax=ax1)
ax1.set_title("PACF Original Data")
plot_pacf(ts_log_shift, ax=ax2)
ax2.set_title("PACF Differentiated Data")
plt.tight_layout()
plt.show()


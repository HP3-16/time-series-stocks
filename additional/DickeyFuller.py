from statsmodels.tsa.stattools import adfuller

def StationarityCheck(timeseries):
    result = adfuller(timeseries['Adj Close'], autolag='AIC')
    print("Results of Dickey Fuller Test")
    print(f'Test Statistics: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Number of lags used: {result[2]}')
    print(f'Number of observations used: {result[3]}')
    for key, value in result[4].items():
        print(f'critical value ({key}): {value}')
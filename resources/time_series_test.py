import matplotlib.pyplot as plot
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox


def plot_rolling(df, lags=60):

    # Determing rolling statistics
    rolmean = df.rolling(lags).mean()
    rolstd = df.rolling(lags).std()

    # Plot rolling statistics:
    plot.figure(figsize=(12, 8))
    plot.plot(df, color='lightseagreen', label='Original')
    plot.plot(rolmean, color='lightcoral', label='Rolling Mean')
    plot.plot(rolstd, color='mediumseagreen', label='Rolling Std')
    plot.legend(loc='best')
    plot.title('Rolling Mean & Standard Deviation')
    plot.show()
    plot.close()


def test_stationarity(df):
    """
    # Perform Dickey-Fuller test:
    if the ‘Test Statistic’ is greater than the ‘Critical Value’ than the time series is stationary.

    Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is
    non-stationary. It has some time dependent structure.
    Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root,
    meaning it is stationary. It does not have time-dependent structure.
    p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
    p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
    """

    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dfoutput[0], dfoutput[4]


def dw_test(error):
    """
    The test statistic is approximately equal to 2*(1-r) where r is the sample autocorrelation of the residuals.
    Thus, for r == 0, indicating no serial correlation, the test statistic equals 2. This statistic will always be
    between 0 and 4. The closer to 0 the statistic, the more evidence for positive serial correlation. The closer to 4,
    the more evidence for negative serial correlation.
    """
    print('dw test', durbin_watson(error, axis=0))


def serial_correlation(error):
    """
    Since no p-value is below .05, both tests agree that you can not reject the null of no auto-correlation between the
    series and each of it's first XX lags with > 95% confidence level.
    """
    # https://robjhyndman.com/hyndsight/ljung-box-test/
    lags = min(60, round(len(error)/5))
    qljb, pvalue = acorr_ljungbox(error, lags=lags)
    print(qljb)
    print(pvalue)

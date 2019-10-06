import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plot
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import normaltest, shapiro
from sklearn import linear_model
import numpy as np

from resources import STRING
from resources import time_series_test

sns.set()

file_list = [filename for filename in os.listdir(STRING.train_processed) if
             filename.endswith('.csv')]

df = pd.read_csv(STRING.train_processed + file_list[0], sep=';', encoding='utf-8')
print(df.columns)

# Customer Number
print(df['UserID'].value_counts())
print(df['UserID'].nunique())

# Number of ads per hour
df['hour'].value_counts().sort_index().plot.line()
plot.xlabel('Hour')
plot.ylabel('Number Ads')
plot.show()

# Relation WInning Bid / Number User sees an Ad
plot.scatter(df['user_id_acumulative'], df['WinningBid'])
plot.xlabel('Number of Ads Seen')
plot.ylabel('Winning Bid')
plot.show()

# Distribution of WinningBid
print(df['WinningBid'].describe())
fig, axs = plot.subplots(ncols=2)
sns.distplot(df['WinningBid'], ax=axs[0])
sns.boxplot(x=df['WinningBid'], ax=axs[1])
plot.show()

# Sum of WinningBid by Minute-Second
df['number_bids'] = pd.Series(1, index=df.index)
ts = df.groupby(['hm_index'], as_index=False)['WinningBid', 'number_bids'].sum()
fig, ax1 = plot.subplots()
_, wft_tend = sm.tsa.filters.hpfilter(ts['WinningBid'])
ts['Trend'] = wft_tend
print(ts)
ax1.plot(ts.index, ts['WinningBid'], label='sum(WinningBid)')
ax1.plot(ts.index, ts['Trend'], label='HP Winning Bid')
plot.ylim(0)
ax1.set_ylabel('Bids', color='r')
ax2 = ax1.twinx()
ax2.plot(ts.index, ts['number_bids'], label='Number Bids', color='mediumseagreen', linestyle='--')
ax2.set_ylabel('User-Ads', color='g')
ax2.grid(False)
ax2.set_ylabel('User-Ads ', color='g')
fig.legend(loc='lower right')
plot.show()
del wft_tend

# Conversion Rate at second level
df.loc[df['WinningBid'] > 0, 'Bidding'] = 1
df_i = df.groupby('hms_index').aggregate({'Bidding': 'sum', 'WinningBid': 'sum', 'number_bids': 'sum'})
df_i = df_i.reset_index(drop=False)
df_i['conversion_rate'] = df_i['Bidding'] / df_i['number_bids'] * 100
print(df_i)
df_i = df_i.sample(100)
fig, axs = plot.subplots(nrows=3)
a = sns.barplot(y='WinningBid', x='hms_index', data=df_i, ax=axs[0], label='Winning Bid')
a.tick_params(labelsize=7, labelbottom=False)
b = sns.barplot(y='conversion_rate', x='hms_index', data=df_i, ax=axs[1], label='Conversion Rate')
b.tick_params(labelsize=7, labelbottom=False)
sns.lineplot(y='number_bids', x='hms_index', data=df_i, ax=axs[2], color='c', label='Number Bids')
sns.lineplot(y='Bidding', x='hms_index', data=df_i, ax=axs[2], color='lightcoral', label='Positive Bid')
plot.show()

# Number of Adds and Users: It can be seen a high correlation between the number of ads showed and the WinningBid
df_sum = df.groupby('hm_index').agg({'number_ads_minute': 'mean', 'number_user_minute': 'mean', 'WinningBid': 'sum'})
df_sum = df_sum.reset_index(drop=False)
print(df_sum)
fig, ax1 = plot.subplots()
ax1.plot(df_sum.index, df_sum['WinningBid'], label='Winning Bid', color='lightcoral')
plot.xticks(fontsize=10, rotation=45)
plot.ylim(0)
ax1.set_ylabel('Bids', color='r')
ax2 = ax1.twinx()
ax2.plot(df_sum['number_ads_minute'], label='Number Ads', color='mediumseagreen')
ax2.plot(df_sum['number_user_minute'], label='Number Users', color='mediumseagreen', linestyle='--')
ax2.set_ylabel('User-Ads', color='g')
fig.legend(loc="lower right")
ax2.grid(False)
plot.ylim(0)
plot.show()

# Autocorrelation by minute
df_sum_second = df.groupby('hms_index').agg(
    {'number_user_second': 'mean', 'number_ads_second': 'mean', 'WinningBid': 'sum'})
df_index = pd.DataFrame(range(0, df['hms_index'].max() + 1, 1), columns=['hms_index'])
df_sum_second = pd.merge(df_index, df_sum_second, how='left', on='hms_index').fillna(0)

df_sum_minute = df.groupby('hm_index').agg(
    {'number_user_minute': 'mean', 'number_ads_minute': 'mean', 'WinningBid': 'sum'})
df_index = pd.DataFrame(range(0, df['hm_index'].max() + 60, 1), columns=['hm_index'])
df_sum_minute = pd.merge(df_index, df_sum_minute, how='left', on='hm_index').fillna(0)

df_sum_minute['AR1_Bid'] = df_sum_minute['WinningBid'].shift(1)
df_sum_minute['AR1_nu'] = df_sum_minute['number_user_minute'].shift(1)
df_sum_minute['AR1_na'] = df_sum_minute['number_ads_minute'].shift(1)

plot.subplot(231)
plot.scatter(x=df_sum_minute['WinningBid'], y=df_sum_minute['AR1_Bid'])
plot.xlabel('Winning Bids')
plot.ylabel('AR1 Winning Bids')

plot.subplot(232)
plot.scatter(x=df_sum_minute['number_user_minute'], y=df_sum_minute['AR1_nu'])
plot.xlabel('Number User Minute')
plot.ylabel('AR1 Number User')

plot.subplot(233)
plot.scatter(x=df_sum_minute['number_ads_minute'], y=df_sum_minute['AR1_na'])
plot.xlabel('Number Ads Minute')
plot.ylabel('AR1 Number Ads')

plot.subplot(234)
plot.scatter(x=df_sum_minute['WinningBid'], y=df_sum_minute['AR1_na'])
plot.xlabel('Number Winning Bids')
plot.ylabel('AR1 Number Ads')

plot.subplot(235)
plot.scatter(x=df_sum_minute['WinningBid'], y=df_sum_minute['AR1_nu'])
plot.xlabel('Number Winning Bids')
plot.ylabel('AR1 Number Users')

plot.subplot(236)
plot.scatter(x=df_sum_minute['number_ads_minute'], y=df_sum_minute['AR1_nu'])
plot.xlabel('Number Ads Minute')
plot.ylabel('AR1 Number User')
plot.show()

del df_index

# Rolling Mean/Variance
time_series_test.plot_rolling(df_sum_second['WinningBid'], lags=60)

# Stationarity Test
time_series_test.test_stationarity(df_sum_second['WinningBid'])

# Autocorrelation
pd.plotting.autocorrelation_plot(df_sum_second['WinningBid'])
plot.show()

fig = plot.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df_sum_second['WinningBid'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df_sum_second['WinningBid'], lags=40, ax=ax2)
plot.show()

# Decompose Time-Series
result_add = seasonal_decompose(df_sum_minute['WinningBid'], model='additive',
                                freq=1)  # Base Level + Trend + Seasonality + Error
result_add.plot().suptitle('Additive Decompose', fontsize=22)
plot.show()
plot.close()

# Browser Types
df['Bidding'] = pd.Series(0, index=df.index)
df.loc[df['WinningBid'] > 0, 'Bidding'] = 1
sns.catplot(y="Browser", hue="Bidding", kind="count",
            palette="pastel", edgecolor=".6",
            data=df)
plot.show()

# Average Winning by Browser
g = sns.catplot(x="Browser", y="WinningBid", data=df,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Average Winning Bid")
plot.show()

# Error Analysis
file_model = linear_model.LinearRegression()
print(df.shape)

df = df.sample(frac=0.5)
y = df['WinningBid'].values
x = df[['number_ads_user', 'hms_index',
        'number_user_second', 'number_ads_second',
        'number_ads_user_second',
        'number_user_minute', 'number_ads_minute',
        'peak6am8am', 'peak14pm16pm', 'user_id_acumulative'] +
       [x for x in df.columns if x.startswith('d_browser')] +
       [x for x in df.columns if x.startswith('ar')] +
       [x for x in df.columns if x.startswith('ma_')]].values
file_model.fit(x, y)
prediction = file_model.predict(x)
error = np.square(y - prediction)

time_series_test.test_stationarity(error)  # stationarity
time_series_test.dw_test(error)  # Durbin-Watson
time_series_test.serial_correlation(error)  # Ljung Box Test

# Plot Residuals
sns.distplot(error)
plot.show()

# Residual Normality
stat, p = shapiro(error)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

stat, p = normaltest(error)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

# Heatmap
df = df[['key_id', 'hms_index', 'hm_index',
         'number_ads_user',
         'number_user_second', 'number_ads_second',
         'number_ads_user_second',
         'number_user_minute', 'number_ads_minute',
         'peak6am8am', 'peak14pm16pm'] +
        [x for x in df.columns if x.startswith('ar')] +
        [x for x in df.columns if x.startswith('ma_')] + ['WinningBid']]
print(df)
corr = df.drop(['key_id', 'hms_index', 'hm_index'], axis=1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plot.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws={"size": 7})
plot.show()

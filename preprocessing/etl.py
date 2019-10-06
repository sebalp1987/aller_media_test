import sys

from resources.spark import SparkJob
from resources import STRING, auxiliar_func

from pyspark.sql.functions import isnan, when, col, to_date, month, dayofweek, dayofmonth, hour, split, \
    concat, minute, lit, count as count_, udf, sum as sum_, approx_count_distinct, max as max_, min as min_, lag, avg, \
    monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType


class EtlJob(SparkJob):

    def __init__(self, train_file=True, ar_lags=10, ar_min_lag=1, ma_ss_lag=None, variable_analysis=True):
        self._spark = self.get_spark_session("etl")
        self.train_file = train_file
        self.ar_lags = ar_lags
        self.ma_ss_lag = ma_ss_lag
        self.ar_min_lag = ar_min_lag
        self.variable_analysis = variable_analysis

    def run(self):
        df, auxiliar_train = self._extract()
        df = self._transform(df, auxiliar_train)
        self._load(df)
        self._spark.stop()

    def _extract(self):
        if self.train_file:
            df = (self._spark.read.csv(STRING.train, header=True, sep=','))
            auxiliar_train = None
        else:
            auxiliar_train = (self._spark.read.csv(STRING.train, header=True, sep=','))
            df = (self._spark.read.csv(STRING.test, header=True, sep=','))

        return df, auxiliar_train

    def _transform(self, df, auxiliar_train):

        if not self.train_file:
            auxiliar_train = auxiliar_train.drop('WinningBid')
            auxiliar_train = auxiliar_train.withColumn('test', lit(0))
            df = df.withColumn('test', lit(1))
            df = auxiliar_train.union(df)
            del auxiliar_train

        # We create the time as Index
        split_col = split(df['ApproximateDate'], ' ')
        df = df.withColumn('time', split_col.getItem(1))  # time

        # Hour Index
        func_index = udf(lambda x: auxiliar_func.time_to_num(x, index='hms'), IntegerType())
        df = df.withColumn('hms_index', func_index(df['time']))

        # We order by UserId-Date
        df = df.orderBy(['UserID', 'hms_index'])

        # We check Null Values
        df.select([count_(when(isnan(c), c)).alias(c) for c in df.columns]).show()

        # We create a rank of users by how many times in the past saw an ad
        w = (Window().partitionBy(df.UserID).orderBy('time').rowsBetween(Window.unboundedPreceding, 0))
        df = df.withColumn('user_id_acumulative', count_(df['UserId']).over(w))

        # Number of Ads/User/Second
        df = df.withColumn('key_id', concat(df['UserID'], lit(' '), df['hms_index']))
        w = (Window().partitionBy(df.key_id).orderBy('hms_index').rowsBetween(-sys.maxsize, sys.maxsize))
        df = df.withColumn('number_ads_user_second', count_(df.key_id).over(w))

        # Number of Ads/User
        df_group = df.groupby(['key_id']).agg(count_('key_id').alias('count_ads'))
        split_col = split(df_group['key_id'], ' ')
        df_group = df_group.withColumn('UserID', split_col.getItem(0))  # time
        w = (Window().partitionBy(df_group.UserID).orderBy('key_id').rowsBetween(Window.unboundedPreceding, 0))
        df_group = df_group.withColumn('number_ads_user', sum_(df_group.count_ads).over(w))
        df_group = df_group.select(['key_id', 'number_ads_user'])
        df = df.join(df_group, how='left', on='key_id')
        del df_group

        # Number of Users/Second
        w = (Window().partitionBy(df.ApproximateDate).rowsBetween(-sys.maxsize, sys.maxsize))
        df = df.withColumn('number_user_second', approx_count_distinct(df.UserID).over(w))

        # Number of Ads/Second
        df = df.withColumn('number_ads_second', count_(df.ApproximateDate).over(w))

        # Browser Dummy Transformation
        types = df.select('Browser').distinct().collect()
        types = [val['Browser'] for val in types]
        new_cols = [when(df['Browser'] == ty, 1).otherwise(0).alias('d_browser_' + ty) for ty in types]
        df = df.select(df.columns + new_cols)

        # Decompose Date Variables
        df = df.withColumn('date', to_date(df['ApproximateDate']))  # date
        df = df.withColumn('month', month(df['ApproximateDate']))  # month
        df = df.withColumn('day', dayofmonth(df['ApproximateDate']))  # day
        df = df.withColumn('weekday', dayofweek(df['ApproximateDate']))  # weekday 1=Monday

        df = df.withColumn('hour', hour(df['time']))  # hour
        df = df.withColumn('minute', minute(df['time']))  # minute

        # Peak Hour
        df = df.withColumn('peak6am8am', when(df['hour'].between(6, 8), 1).otherwise(0))
        df = df.withColumn('peak14pm16pm', when(df['hour'].between(14, 16), 1).otherwise(0))

        # Minute Index
        func_index = udf(lambda x: auxiliar_func.time_to_num(x, index='hm'), IntegerType())
        df = df.withColumn('hm_index', func_index(df['time']))

        # Convert to time-series by Minute
        # We reduce to minutes
        df_time_serie_ads = df.select(
            ['hms_index', 'hm_index', 'number_user_second', 'number_ads_second']).drop_duplicates()
        df_time_serie_user = df.select(['UserID', 'hm_index']).drop_duplicates()

        # Group-by the values
        df_time_serie_user = df_time_serie_user.groupBy('hm_index').agg(approx_count_distinct('UserID'))
        df_time_serie_ads = df_time_serie_ads.groupBy('hm_index').agg(
            {'number_ads_second': 'sum'}).drop_duplicates(
            subset=['hm_index'])

        # Join ads-users per minute
        df_time_serie = df_time_serie_ads.join(df_time_serie_user, how='left', on='hm_index')
        del df_time_serie_ads, df_time_serie_user

        # Rename columns
        df_time_serie = df_time_serie.withColumnRenamed('sum(number_ads_second)',
                                                        'number_ads_minute').withColumnRenamed(
            'approx_count_distinct(UserID)',
            'number_user_minute')

        # Resample Range of Minutes
        resample_range = list(range(df_time_serie.select(min_(col('hm_index'))).limit(1).collect()[0][0],
                                    df_time_serie.select(max_(col('hm_index'))).limit(1).collect()[0][0] + 1, 1))

        resample_range = self._spark.createDataFrame(resample_range, IntegerType())

        # Join the original df
        df_time_serie = resample_range.join(df_time_serie, how='left',
                                            on=resample_range.value == df_time_serie.hm_index).drop(
            *['hm_index']).fillna(0)

        # Create Lags By Minutes
        w = Window().partitionBy().orderBy(col('value'))
        if self.ar_min_lag > 0:
            df_time_serie = df_time_serie.select('*', lag('number_user_minute').over(w).alias('ar1_number_user_minute'))
            df_time_serie = df_time_serie.select('*', lag('number_ads_minute').over(w).alias('ar1_number_ads_minute'))

            if self.ar_min_lag > 1:
                for l in range(2, self.ar_min_lag + 1, 1):
                    df_time_serie = df_time_serie.select('*',
                                                         lag('ar' + str(l - 1) + '_number_user_minute').over(w).alias(
                                                             'ar' + str(l) + '_number_user_minute'))
                    df_time_serie = df_time_serie.select('*',
                                                         lag('ar' + str(l - 1) + '_number_ads_minute').over(w).alias(
                                                             'ar' + str(l) + '_number_ads_minute'))

        # Remove the lagged Null Values
        df_time_serie = df_time_serie.dropna()

        # join and remove lag Null values of the first minute
        df = df.orderBy(['UserID', 'hms_index'])
        df = df.join(df_time_serie.orderBy(['hm_index']), how='left', on=df.hm_index == df_time_serie.value).drop(
            'value')

        # Convert to time-series and resample by Seconds
        df_time_serie = df.select(['hms_index', 'number_user_second', 'number_ads_second']).drop_duplicates()
        resample_range = list(range(df_time_serie.select(min_(col('hms_index'))).limit(1).collect()[0][0],
                                    df_time_serie.select(max_(col('hms_index'))).limit(1).collect()[0][0] + 1, 1))
        resample_range = self._spark.createDataFrame(resample_range, IntegerType())

        # Join the original df
        df_time_serie = resample_range.join(df_time_serie, how='left',
                                            on=resample_range.value == df_time_serie.hms_index).drop(
            *['hms_index']).fillna(0)

        # Create lags
        w = Window().partitionBy().orderBy(col('value'))
        if self.ar_lags > 0:
            df_time_serie = df_time_serie.select('*', lag('number_user_second').over(w).alias('ar1_number_user_second'))
            df_time_serie = df_time_serie.select('*', lag('number_ads_second').over(w).alias('ar1_number_ads_second'))

            if self.ar_lags > 1:
                for l in range(2, self.ar_lags + 1, 1):
                    df_time_serie = df_time_serie.select('*',
                                                         lag('ar' + str(l - 1) + '_number_user_second').over(w).alias(
                                                             'ar' + str(l) + '_number_user_second'))
                    df_time_serie = df_time_serie.select('*',
                                                         lag('ar' + str(l - 1) + '_number_ads_second').over(w).alias(
                                                             'ar' + str(l) + '_number_ads_second'))

        # Create Moving Average
        if self.ma_ss_lag is not None:

            # Get hour from index
            func_index = udf(lambda x: auxiliar_func.num_to_time(x), StringType())
            df_time_serie = df_time_serie.withColumn('time', func_index(df_time_serie['value']))

            # minute MA terms (Average per second last xx seconds)
            if self.ma_ss_lag is not None:
                for lag_val in self.ma_ss_lag:
                    # range to take into account
                    w = (
                        Window.orderBy(df_time_serie['value']).rangeBetween(
                            -lag_val, 0))
                    # MA variables
                    df_time_serie = df_time_serie.withColumn('ma_seconds_' + str(lag_val) + '_number_user_second',
                                                             avg('number_user_second').over(w))
                    df_time_serie = df_time_serie.withColumn('ma_seconds_' + str(lag_val) + '_number_ads_second',
                                                             avg('number_ads_second').over(w))

                    # Increasing ID
                    df_time_serie = df_time_serie.withColumn('rn', monotonically_increasing_id())

                    # Replace first values by Null
                    df_time_serie = df_time_serie.withColumn(
                        'ma_seconds_' + str(lag_val) + '_number_user_second',
                        when(df_time_serie['rn'] < lag_val, None).otherwise(
                            df_time_serie['ma_seconds_' + str(lag_val) + '_number_user_second']))

                    df_time_serie = df_time_serie.withColumn(
                        'ma_seconds_' + str(lag_val) + '_number_ads_second',
                        when(df_time_serie['rn'] < lag_val, None).otherwise(
                            df_time_serie['ma_seconds_' + str(lag_val) + '_number_ads_second']))

                    # Get the average by Minute
                    df_time_serie = df_time_serie.withColumn('ma_minute_' + str(lag_val) + '_number_user_second',
                                                             df_time_serie['ma_seconds_' + str(
                                                                 lag_val) + '_number_user_second'] * 60)
                    df_time_serie = df_time_serie.withColumn('ma_minute_' + str(lag_val) + '_number_ads_second',
                                                             df_time_serie['ma_seconds_' + str(
                                                                 lag_val) + '_number_ads_second'] * 60)
                df_time_serie = df_time_serie.drop(*['rn'])

        # Remove the lagged Null Values
        df_time_serie = df_time_serie.drop(*['time', 'number_user_second', 'number_ads_second']).dropna()
        # join and remove lag Null values of the first minute
        df = df.join(df_time_serie.orderBy(['value']), how='left', on=df.hms_index == df_time_serie.value).drop(
            'value').dropna()

        if self.train_file and not self.variable_analysis:
            df = df.select(['key_id', 'hms_index',
                            'number_ads_user',
                            'number_user_second', 'number_ads_second',
                            'number_ads_user_second',
                            'peak6am8am', 'peak14pm16pm', 'user_id_acumulative'] +
                           [x for x in df.columns if x.startswith('d_browser')] +
                           [x for x in df.columns if x.startswith('ar')] +
                           [x for x in df.columns if x.startswith('ma_')] + ['WinningBid'])

        if not self.train_file:
            df = df.filter(df['test'] == 1)
            df = df.select(['key_id',
                            'number_ads_user', 'hms_index',
                            'number_user_second', 'number_ads_second',
                            'number_ads_user_second',
                            'peak6am8am', 'peak14pm16pm', 'user_id_acumulative'] +
                           [x for x in df.columns if x.startswith('d_browser')] +
                           [x for x in df.columns if x.startswith('ar')] +
                           [x for x in df.columns if x.startswith('ma_')])

        df = df.orderBy(['hms_index', 'UserID'])
        df.show()
        return df

    def _load(self, df):
        if self.train_file and self.variable_analysis:
            df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(
                STRING.train_processed)
        elif self.train_file and not self.variable_analysis:
            df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(
                STRING.train_model)
        elif not self.train_file:
            df.coalesce(1).write.mode("overwrite").option("header", "true").option("sep", ";").csv(
                STRING.test_model)


if __name__ == '__main__':
    EtlJob(train_file=True, ar_lags=10, ar_min_lag=1, ma_ss_lag=[60], variable_analysis=True).run()

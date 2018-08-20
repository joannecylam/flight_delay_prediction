import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler


class PrepareData(object):

    def __init__(self, filename="flight_delays_data.csv"):
        self.cnt_var = ['Week', 'std_hour', 'delay_time', 'is_claim']
        self.bin_var = ['Arrival', 'Airline']
        self.drop_var = ['Departure', 'flight_no']
        self.filename = filename

    def normalize_df(self, df):
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = scaler.fit_transform(df.values)
        return pd.DataFrame(values, columns=df.columns.tolist(), index=df.index)

    def cvt_bin(self, df):
        for _var in self.bin_var:
            print "converting bin var:", _var
            for item in df[_var].unique():
                try:
                    field_name = _var + "_" + item
                    df[field_name] = df[[_var]] == item
                except Exception as e:
                    print e
                    print "value name:", item
        return df

    def cvt_delay_time(self, df):
        times = []
        for t in df['delay_time']:
            if t == "Cancelled":
                times.append(100)
            else:
                times.append(float(t))
        df['delay_time'] = times
        self.delay_min = min(times)
        self.delay_max = max(times)
        return df

    def load_raw_data(self, additional_kwargs, time_series):
        kwargs = {}
        if time_series:
            kwargs.update({
                'parse_dates': {"dt": ['flight_date']},
                'infer_datetime_format': True,
                'index_col': 'dt'
            })
        else:
            kwargs.update({'index_col': 'flight_id'})

        kwargs.update(additional_kwargs)
        df = pd.read_csv(
            self.filename,
            na_values=['NaN', '?','nan'],
            **kwargs)
        return df

    def clean_df(self, df):
        df.drop(self.drop_var + self.bin_var, axis=1, inplace=True)
        df = self.cvt_delay_time(df)
        return df

    def load_data(self, additional_kwargs={}, time_series=False):
        df = self.load_raw_data(additional_kwargs, time_series)
        df = self.cvt_bin(df)
        df = self.clean_df(df)
        df = self.cvt_datetime(df)
        df = self.normalize_df(df)
        return df

    def cvt_datetime(self, df, dt_label="flight_date"):
        flight_dates = [datetime.datetime.strptime(str_dt, '%Y-%m-%d').date() for str_dt in df['flight_date'].values]
        df['flight_year'] = [dt.year for dt in flight_dates]
        df['flight_month'] = [dt.month for dt in flight_dates]
        df['flight_day'] = [dt.day for dt in flight_dates]
        df.drop(dt_label, axis=1, inplace=True)
        return df

    def build_train(self, df, label="delay_time"):
        return df.drop('delay_time', axis=1), df['delay_time']

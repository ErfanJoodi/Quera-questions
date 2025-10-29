import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.customer_features = None

    def _parse_datetime(self, col='DateTime_CartFinalize'):
        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        self.df['hour'] = self.df[col].dt.hour.fillna(-1).astype(int)
        self.df['dayofweek'] = self.df[col].dt.dayofweek.fillna(-1).astype(int)
        self.df['month'] = self.df[col].dt.month.fillna(0).astype(int)
        self.df['is_weekend'] = self.df['dayofweek'].isin([5,6]).astype(int)

    def _handle_missing(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.df[num_cols] = self.df[num_cols].fillna(0)
        obj_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.df[obj_cols] = self.df[obj_cols].fillna('missing')

    def _freq_encode(self, col='city_name_fa'):
        freq = self.df[col].value_counts(dropna=False)
        self.df[f'{col}_freq'] = self.df[col].map(freq).fillna(0).astype(int)

    def _basic_transforms(self):
        self.df['Amount_Gross_Order_log'] = np.log1p(self.df['Amount_Gross_Order'].astype(float).clip(lower=0))
        self.df['Quantity_item_log'] = np.log1p(self.df['Quantity_item'].astype(float).clip(lower=0))

    def _customer_aggregations(self):
        if 'DateTime_CartFinalize' in self.df.columns and self.df['DateTime_CartFinalize'].notna().any():
            max_dt = self.df['DateTime_CartFinalize'].max()
            self.df['days_from_max'] = (max_dt - self.df['DateTime_CartFinalize']).dt.days.fillna(9999).astype(int)
        else:
            self.df['days_from_max'] = 9999

        agg_funcs = {
            'ID_Order': 'count',
            'Amount_Gross_Order': ['sum', 'mean', 'std', 'max', 'min'],
            'Amount_Gross_Order_log': ['mean', 'std'],
            'Quantity_item': ['sum', 'mean'],
            'Quantity_item_log': ['mean'],
            'hour': ['nunique', 'mean'],
            'dayofweek': ['nunique'],
            'city_name_fa_freq': ['mean'],
            'days_from_max': ['min', 'mean']
        }

        grouped = self.df.groupby('ID_Customer').agg(agg_funcs)
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped['orders_per_customer'] = grouped['ID_Order_count']
        recency_flag = self.df['days_from_max'] <= 30
        recent_counts = self.df[recency_flag].groupby('ID_Customer').size().rename('recent_count')
        grouped = grouped.join(recent_counts, how='left').fillna(0)
        grouped['recent_frac'] = grouped['recent_count'] / (grouped['orders_per_customer'].replace(0, np.nan))
        grouped['recent_frac'] = grouped['recent_frac'].fillna(0)
        grouped = grouped.fillna(0)
        keep_cols = [c for c in grouped.columns if not c.startswith('ID_Order')]
        self.customer_features = grouped[keep_cols].copy()

    def _scale_customer_features(self):
        scaler = StandardScaler()
        numeric = self.customer_features.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) == 0:
            return
        self.customer_features[numeric] = scaler.fit_transform(self.customer_features[numeric])

    def transform(self):
        self._handle_missing()
        self._parse_datetime()
        self._freq_encode('city_name_fa')
        self._basic_transforms()
        self._customer_aggregations()
        self._scale_customer_features()
        return self.df, self.customer_features

import pandas as pd
import logging as log

from numpy import number


class DataConverter:

    def _to_celcius(self, temp_f: float):
        return round((temp_f - 32) / 1.8, 1)

    def fahrenheit_to_celcius(self, df: pd.DataFrame, feature_name: str, target_feature_name: str = None, remove_old_feature:bool = False):
        if target_feature_name is None:
            target_feature_name = feature_name
        df[target_feature_name] = df[feature_name].apply(self._to_celcius)
        log.debug(f'{target_feature_name}:\n{df[target_feature_name].describe()}')
        if remove_old_feature:
            df = df.drop(columns=[feature_name])
        return df

    def encode_categorial_with_map(self, df: pd.DataFrame, feature_name:str, conversion_map: dict):
        df[feature_name] = df[feature_name].map(conversion_map).astype(int)
        return df

    def _get_age_group(self, age:number):
        if age < 18:
            return 'Under 18'
        elif age >= 18 and age < 25:
            return '18-24'
        elif age >= 25 and age < 35:
            return '25-34'
        elif age >= 35 and age < 45:
            return '35-44'
        elif age >= 45 and age < 55:
            return '45-54'
        elif age >= 55 and age < 65:
            return '55-64'
        else:
            return '65+'

    def group_by_age(self, df: pd.DataFrame, feature_name:str, target_feature_name:str = None, remove_old_feature:bool = False):
        if target_feature_name is None:
            target_feature_name = feature_name
        df[target_feature_name] = df[feature_name].apply(self._get_age_group)
        log.debug(f'{target_feature_name}:\n{df[target_feature_name].describe()}')
        if remove_old_feature:
            df = df.drop(columns=[feature_name])
        return df

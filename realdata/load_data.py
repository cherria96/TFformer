import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from omegaconf import ListConfig
import json

#from src import ROOT_PATH

logger = logging.getLogger(__name__)


def process_static_features(static_features: pd.DataFrame, drop_first=False) -> pd.DataFrame:
    """
    Global standard normalisation of static features & one hot encoding
    Args:
        static_features: pd.DataFrame with unprocessed static features
        drop_first: Dropping first class of one-hot-encoded features

    Returns: pd.DataFrame with pre-processed static features

    """
    processed_static_features = []
    for feature in static_features.columns:
        if isinstance(static_features[feature].iloc[0], float):
            mean = np.mean(static_features[feature])
            std = np.std(static_features[feature])
            processed_static_features.append((static_features[feature] - mean) / std)
        else:
            one_hot = pd.get_dummies(static_features[feature], drop_first=drop_first)
            processed_static_features.append(one_hot.astype(float))

    static_features = pd.concat(processed_static_features, axis=1)
    return static_features


def load_weather_data_processed(data_path: str,
                               min_seq_length: int = None,
                               max_seq_length: int = None,
                               treatment_list: List[str] = None,
                               outcome_list: List[str] = None,
                               vital_list: List[str] = None,
                               static_list: List[str] = None,
                               max_number: int = None,
                               data_seed: int = 100,
                               drop_first=False,
                               **kwargs) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict):
    """
    Load and pre-process weather 10 minutes periodic dataset (for real-world experiments)
    :param data_path: Path with weather dataset 
    :param min_seq_length: Min sequence lenght in cohort
    :param min_seq_length: Max sequence lenght in cohort
    :param treatment_list: List of treaments
    :param outcome_list: List of outcomes
    :param vital_list: List of vitals (time-varying covariates)
    :param static_list: List of static features
    :param max_number: Maximum number of patients in cohort
    :param data_seed: Seed for random cohort patient selection
    :param drop_first: Dropping first class of one-hot-encoded features
    :return: tuple of DataFrames and params (treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params)
    """

    logger.info(f'Loading weather dataset from {data_path}.')

    # h5 = pd.HDFStore(data_path, 'r')
    h5 = pd.read_csv(data_path)[143:143+144*100]
    # # Save with new colums name
    # new_columns = ['date','p','T','Tpot','Tdew','rh','VPmax','VPact','VPdef','sh','H2OC','rho','wv','maxwv','wd','rain','raining','SWDR','PAR','maxPAR','Tlog','OT']
    # h5.columns = new_columns
    # h5[['date','time']] = h5['date'].str.split(expand = True)
    # h5.to_csv('weather/weather_.csv')
    # The weather data hs following features. All of data type would be float maybe... except data(yyyy-mm-dd hh:mm:ss)
    # date', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    # 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    # 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
    # 'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m�)',
    # 'PAR (�mol/m�/s)', 'max. PAR (�mol/m�/s)', 'Tlog (degC)', 'OT'

    if treatment_list is None:
        treatment_list = [
                          'p',
                          #'T',
                          'H2OC',
                          'Tpot',
                          'Tdew']
    if outcome_list is None:
        outcome_list = [
            'T'
            #'rh',
            #'VPmax',
            #'VPact'
            # 'VPdef',
            # 'sh',
            # 'H2OC',
            # 'rho'
        ]
    else:
        outcome_list = ListConfig([outcome.replace('_', ' ') for outcome in outcome_list])
    if vital_list is None:
        vital_list = [
            #'wv','maxwv'#,'wd','rain'
        ]
    if static_list is None:
        static_list = [
            'raining'#,'SWDR','PAR','maxPAR','Tlog','OT'
        ]
    
    # treatments = h5['/interventions'][treatment_list]
    # all_vitals = h5['/vitals_labs_mean'][outcome_list + vital_list]
    # static_features = h5['/patients'][static_list]
    treatments = h5[treatment_list]
    treatments.set_index(h5['date'],inplace = True)
    all_vitals = h5[outcome_list + vital_list]
    all_vitals.set_index(h5['date'],inplace = True)
    static_features = h5[static_list]
    static_features.set_index(h5['date'],inplace = True)
    # treatments = treatments.droplevel(['hadm_id', 'icustay_id'])
    # all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
    column_names = []
    for column in all_vitals.columns:
        if isinstance(column, str):
            column_names.append(column)
        else:
            column_names.append(column[0])
    all_vitals.columns = column_names
    # static_features = static_features.droplevel(['hadm_id', 'icustay_id'])
    # Filling NA
    all_vitals = all_vitals.fillna(method='ffill')
    all_vitals = all_vitals.fillna(method='bfill')
    # Filtering longer then min_seq_length and cropping to max_seq_length
    user_sizes = all_vitals.groupby('date').size()
    filtered_users = user_sizes.index[user_sizes >= min_seq_length] if min_seq_length is not None else user_sizes.index
    if max_number is not None:
        np.random.seed(data_seed)
        filtered_users = np.random.choice(filtered_users, size=max_number, replace=False)
    treatments = treatments.loc[filtered_users]
    all_vitals = all_vitals.loc[filtered_users]
    if max_seq_length is not None:
        treatments = treatments.groupby('date').head(max_seq_length)
        all_vitals = all_vitals.groupby('date').head(max_seq_length)
    static_features = static_features[static_features.index.isin(filtered_users)]
    logger.info(f'Number of patients filtered: {len(filtered_users)}.')
    # Global scaling (same as with semi-synthetic)
    outcomes_unscaled = all_vitals[outcome_list].copy()
    mean = np.mean(all_vitals, axis=0)
    std = np.std(all_vitals, axis=0)
    all_vitals = (all_vitals - mean) / std

    # Splitting outcomes and vitals
    outcomes = all_vitals[outcome_list].copy()

    vitals = all_vitals[vital_list].copy()
    static_features = process_static_features(static_features, drop_first=False)
    scaling_params = {
        'output_means': mean[outcome_list].to_numpy(),
        'output_stds': std[outcome_list].to_numpy(),
    }
    #print(treatments)
    #print(vitals)
    #print(outcomes)
    #print(static_features)
    return treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params


def load_var4_data_processed(data_path: str,
                               min_seq_length: int = None,
                               max_seq_length: int = None,
                               dim_treatment: int = None,
                               dim_covariates: int = None,
                               dim_outcome: int = None,
                               dim_static: int = None,
                               max_number: int = None,
                               data_seed: int = 100,
                               drop_first=False,
                               **kwargs) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict):
    """
    Load and pre-process var synthetic dataset
    :param data_path: Path with weather dataset 
    :param min_seq_length: Min sequence lenght in cohort
    :param min_seq_length: Max sequence lenght in cohort
    :param treatment_list: List of treaments
    :param outcome_list: List of outcomes
    :param vital_list: List of vitals (time-varying covariates)
    :param static_list: List of static features
    :param max_number: Maximum number of patients in cohort
    :param data_seed: Seed for random cohort patient selection
    :param drop_first: Dropping first class of one-hot-encoded features
    :return: tuple of DataFrames and params (treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params)
    """

    logger.info(f'Loading var synthesized dataset from {data_path}.')
    h5 = pd.HDFStore(data_path, 'r')
    treatment_list = [f'T{n}' for n in range(dim_treatment)]
    covariates_list = [f'C{n}' for n in range(dim_covariates)]
    outcome_list = [f'O{n}' for n in range(dim_outcome)]
    static_list = [f'S{n}' for n in range(dim_static)]

    treatments = h5['/treatments'][treatment_list]
    all_vitals = h5['/covariates'][outcome_list + covariates_list]
    static_features = h5['/static_features'][static_list]

    # Global scaling (same as with semi-synthetic)
    outcomes_unscaled = all_vitals[outcome_list].copy()
    mean = np.mean(all_vitals, axis=0)
    std = np.std(all_vitals, axis=0)
    all_vitals = (all_vitals - mean) / std

    # Splitting outcomes and vitals
    outcomes = all_vitals[outcome_list].copy()

    vitals = all_vitals[covariates_list].copy()
    static_features = process_static_features(static_features, drop_first=drop_first)
    scaling_params = {
        'output_means': mean[outcome_list].to_numpy(),
        'output_stds': std[outcome_list].to_numpy(),
    }
    h5.close()
    return treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params


if __name__ == "__main__":
    #data_path = 'weather/weather_.csv
    data_path = '../synthetic-data/data/VAR4.h5'
    treatments, outcomes, vitals, stat_features, outcomes_unscaled, scaling_params = \
        load_var4_data_processed(data_path, min_seq_length=None, max_seq_length=None, dim_covariates=5, dim_outcome=1, dim_static=2, dim_treatment=2)
import os
import pandas as pd
import numpy as np


def join_met_datasets():
    """
    A small script that takes all of the individual hourly datasets and combines them into one dataset.

    WARNING: These small scripts are just for one time uses to generate the datasets that this project
    ultimately uses. These scripts should not be called unless the data needs to be manually regenerated.

    An additional warning for this particular script is that it takes a long time to run and combline all of
    the datasets into one.

    READ THE README IN data/datasets/noaa_mlo/met FOR MORE INFORMATION ON THE NATURE OF THIS DATA.
    """
    rel_path = 'data/datasets/noaa_mlo/met'
    features = ['site', 'year', 'month', 'day', 'hour', 'wind_direction', 'wind_speed', 'wind_steadiness', 'pressure',
                'temp2m', 'temp10m', 'temp_tower', 'rel_humidity', 'precipitation_intensity']
    mlo_met_datasets = [file for file in os.listdir(rel_path) if file.endswith('.txt')]

    df = pd.DataFrame(columns=features)
    i = 0
    for dataset in mlo_met_datasets:
        with open(os.path.join(rel_path, dataset)) as fp:
            for a in [line.split() for line in fp.read().split('\n')]:
                print(a)
                if len(a) == len(features):
                    df.loc[i] = a
                    i += 1
    df.to_csv(os.path.join(rel_path, 'mlo_ytd_hourly.csv'), index=False)


def create_monthly_met_mean_dataset():
    """
    A small script that takes all of the hourly data and turns it into a monthly mean dataset.
    The mean of all averages are taken for every month of each year in the dataset and appended
    to the overall dataset of monthly mean meteorological data.

    WARNING: These small scripts are just for one time uses to generate the datasets that this project
    ultimately uses. These scripts should not be called unless the data needs to be manually regenerated.

    READ THE README IN data/datasets/noaa_mlo/met FOR MORE INFORMATION ON THE NATURE OF THIS DATA.
    """
    rel_path = 'data/datasets/noaa_mlo/met'
    features = ['site', 'year', 'month', 'wind_direction', 'wind_speed', 'wind_steadiness', 'pressure',
                 'temp2m', 'temp10m', 'temp_tower', 'rel_humidity', 'precipitation_intensity']
    df = pd.read_csv(os.path.join(rel_path, 'mlo_ytd_hourly.csv'), na_values=[-999.9, -99])

    mo_mean_dataset = pd.DataFrame(columns=features)
    for year in range(1977, 2021):
        for month in range(1, 13):
            mo: pd.DataFrame = df.loc[df['year'] == year].loc[df['month'] == month]
            mo_mean: pd.DataFrame = mo[['wind_direction', 'wind_speed', 'wind_steadiness', 'pressure', 'temp2m', 'temp10m', 'temp_tower', 'rel_humidity', 'precipitation_intensity']].mean()
            t = pd.DataFrame(columns=features)
            t = t.append(mo_mean, ignore_index=True)
            t['site'] = 'MLO'
            t['year'] = year
            t['month'] = month
            mo_mean_dataset = mo_mean_dataset.append(t, ignore_index=True)
    mo_mean_dataset.to_csv(os.path.join(rel_path, 'mlo_ytd_meanavg.csv'), index=False)


def create_flask_dataset():
    """
    Combines all of the monthly flask datasets into one dataset.

    WARNING: These small scripts are just for one time uses to generate the datasets that this project
    ultimately uses. These scripts should not be called unless the data needs to be manually regenerated.

    READ THE READMEs IN data/datasets/noaa_mlo/ccg/flasks FOR MORE INFORMATION ON THE NATURE OF THIS DATA.
    """
    rel_path = 'data/datasets/noaa_mlo/ccgg/flasks'
    co2_dataset = 'co2_mlo_surface-flask_1_ccgg_month.txt'
    ch4_dataset = 'ch4_mlo_surface-flask_1_ccgg_month.txt'
    n2o_dataset = 'n2o_mlo_surface-flask_1_ccgg_month.txt'
    sf6_dataset = 'sf6_mlo_surface-flask_1_ccgg_month.txt'
    individual_features = ['site', 'year', 'month', 'value']
    co2_df = pd.read_csv(os.path.join(rel_path, co2_dataset), skiprows=70, names=individual_features, sep='\s+')
    ch4_df = pd.read_csv(os.path.join(rel_path, ch4_dataset), skiprows=70, names=individual_features, sep='\s+')
    n2o_df = pd.read_csv(os.path.join(rel_path, n2o_dataset), skiprows=70, names=individual_features, sep='\s+')
    sf6_df = pd.read_csv(os.path.join(rel_path, sf6_dataset), skiprows=70, names=individual_features, sep='\s+')
    co2_df.rename({'value': 'co2_mean'}, axis=1, inplace=True)
    ch4_df.rename({'value': 'ch4_mean'}, axis=1, inplace=True)
    n2o_df.rename({'value': 'n2o_mean'}, axis=1, inplace=True)
    sf6_df.rename({'value': 'sf6_mean'}, axis=1, inplace=True)
    atmosphere_df = pd.merge(pd.merge(pd.merge(co2_df, ch4_df, how='outer'), n2o_df, how='outer'), sf6_df, how='outer')
    atmosphere_df.to_csv(os.path.join(rel_path, 'mlo_atmospheric_flask_monthly_means.csv'), index=False)


def create_full_dataset():
    rel_path = 'data/datasets/noaa_mlo'
    atm_rel_path = 'data/datasets/noaa_mlo/ccgg/flasks'
    atmosphere_df = pd.read_csv(os.path.join(atm_rel_path, 'mlo_atmospheric_flask_monthly_means.csv'))
    met_rel_path = 'data/datasets/noaa_mlo/met'
    meterological_df = pd.read_csv(os.path.join(met_rel_path, 'mlo_ytd_meanavg.csv'))
    mlo_full_df = pd.merge(atmosphere_df, meterological_df, how='outer')
    mlo_full_df.to_csv(os.path.join(rel_path, 'mlo_full.csv'), index=False)


if __name__ == '__main__':
    # join_met_datasets()
    # create_monthly_met_mean_dataset()
    # create_flask_dataset()
    create_full_dataset()

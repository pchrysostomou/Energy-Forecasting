import pandas as pd # new
import numpy as np
import matplotlib.pyplot as plt # new
import os

def get_df_EDA(start, end):
    '''
    Function to get the target data and do seasonal indicator and seasonal encodings

    parameters:
    start   int. Starting year
    end     int. Ending year

    Steps:
    1. read csv to df
    2. join the df of each year
    3. get seasonal indicator: day_of_week, day_of_year --> store to df
    5. return df
    '''
    data_dir = os.path.join(os.path.dirname(__file__), '../data/processed/data')
    df = pd.read_csv(os.path.join(data_dir, f"demanddata_{start}.csv"))
    df = df[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]]
    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])

    if start!= end:
        for i in range(start+1, end+1):
            df1 = pd.read_csv(os.path.join(data_dir, f"demanddata_{i}.csv"))
            df1 = df1[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]]
            df1['SETTLEMENT_DATE'] = pd.to_datetime(df1['SETTLEMENT_DATE'])
            df = pd.concat([df, df1], ignore_index=True)
        
    # seasonal indicator
    df['day_of_week'] = df['SETTLEMENT_DATE'].dt.dayofweek
    df['day_of_year'] = df['SETTLEMENT_DATE'].dt.dayofyear

    return df



if __name__ == '__main__':

    # show the yearly trend
    df = get_df_EDA(2009, 2025)
    df.plot(x='SETTLEMENT_DATE', y='ND', figsize=(10,6))
    plt.title("National Electricity Demand Between 2009 - 2025")
    plt.xlabel("SETTLEMENT DATE")
    plt.ylabel("Electricity Demand(MW)")
    plt.savefig('.././figures/yearly_EDA.png')

    # # show the seasonal trend by only selecting 2 years to investigate
    df = get_df_EDA(2012, 2014)
    df.plot(x='SETTLEMENT_DATE', y='ND', figsize=(10,6))
    plt.title("National Electricity Demand Between 2012 - 2014")
    plt.xlabel("SETTLEMENT DATE")
    plt.ylabel("Electricity Demand(MW)")
    plt.savefig('.././figures/seasonal_EDA.png')

    # # show the weekly trend by only selecting 4 months to investigate 
    # # 2 Adjacent monthes from highest ND, and 2 Adjacent monthes from lowest ND
    df = get_df_EDA(2014, 2014)
    df[df['SETTLEMENT_DATE'].dt.month.isin([11, 12]) ].plot(x='SETTLEMENT_DATE', y='ND', figsize=(10,6))
    plt.title("National Electricity Demand Between Nov to Dec in 2014")
    plt.xlabel("SETTLEMENT DATE")
    plt.ylabel("Electricity Demand(MW)")
    plt.savefig('.././figures/weekly_EDA1.png')
    df[df['SETTLEMENT_DATE'].dt.month.isin([7, 8])].plot(x='SETTLEMENT_DATE', y='ND', figsize=(10,6))
    plt.title("National Electricity Demand Between Jul to Aug in 2014")
    plt.xlabel("SETTLEMENT DATE")
    plt.ylabel("Electricity Demand(MW)")
    plt.savefig('.././figures/weekly_EDA2.png')

    # # select the date with highest ND in Dec, and the date with lowest ND in Aug and their Adjacent dates
    # # To investigate daily trend
    df1 = df[(df['SETTLEMENT_DATE'].dt.month == 12) & (df['SETTLEMENT_DATE'].dt.day.isin([3, 4, 5])) ]
    df1 = df1.pivot(index='SETTLEMENT_PERIOD', columns='SETTLEMENT_DATE', values='ND')
    df2 = df[(df['SETTLEMENT_DATE'].dt.month == 8) & (df['SETTLEMENT_DATE'].dt.day.isin([16, 17, 18])) ]
    df2 = df2.pivot(index='SETTLEMENT_PERIOD', columns='SETTLEMENT_DATE', values='ND')
    df2 = pd.concat([df1, df2])
    df2.plot(figsize=(10,6))
    plt.title("National Electricity Demand")
    plt.xlabel("SETTLEMENT PERIOD")
    plt.ylabel("Electricity Demand(MW)")
    plt.savefig('.././figures/daily_EDA.png')
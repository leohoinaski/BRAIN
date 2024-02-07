#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:48:15 2024

@author: leohoinaski
"""
#import os
import numpy as np
from datetime import datetime
import pandas as pd
import netCDF4 as nc
#from numpy.lib.stride_tricks import sliding_window_view
#import pyproj
#from shapely.geometry import Point
#import geopandas as gpd
#from ismember import ismember
#import wrf

def datePrepBRAIN(ds):
    tf = np.array(ds['TFLAG'][:])
    date=[]
    for ii in range(0,tf.shape[0]):
        date.append(datetime.strptime(tf[:].astype(str)[ii], '%Y%m%d%H').strftime('%Y-%m-%d %H:00:00'))
    
    date = np.array(date,dtype='datetime64[s]')
    dates = pd.DatetimeIndex(date)
    datesTime=pd.DataFrame()
    datesTime['year'] = dates.year
    datesTime['month'] = dates.month
    datesTime['day'] = dates.day
    datesTime['hour'] = dates.hour
    datesTime['datetime']=dates
    return datesTime

def fixTimeBRAIN(ds,data):
    dd = datePrepBRAIN(ds)
    idx2Remove = np.array(dd.drop_duplicates().index)
    data = data[idx2Remove]
    datesTime = dd.drop_duplicates().reset_index(drop=True)
    return datesTime,data

def fixTimeBRAINemis(ds,data):
    dd = datePrepBRAINemis(ds)
    idx2Remove = np.array(dd.drop_duplicates().index)
    data = data[idx2Remove]
    datesTime = dd.drop_duplicates().reset_index(drop=True)
    return datesTime,data

def datePrepBRAINemis(ds):
    tf = np.array(ds['TFLAG'][0:8759])
    date=[]
    for ii in range(0,tf.shape[0]):
        date.append(datetime.strptime(tf[:].astype(str)[ii], '%Y%m%d%H').strftime('%Y-%m-%d %H:00:00'))
    
    date = np.array(date,dtype='datetime64[s]')
    dates = pd.DatetimeIndex(date)
    datesTime=pd.DataFrame()
    datesTime['year'] = dates.year
    datesTime['month'] = dates.month
    datesTime['day'] = dates.day
    datesTime['hour'] = dates.hour
    datesTime['datetime']=dates
    return datesTime

def dailyAverage (datesTime,data):
    if len(data.shape)>3:
        daily = datesTime.groupby(['year','month','day']).count()
        dailyData = np.empty((daily.shape[0],data.shape[1],data.shape[2],data.shape[3]))
        for day in range(0,daily.shape[0]):
            findArr = (datesTime['year'] == daily.index[day][0]) & \
                (datesTime['month'] == daily.index[day][1]) & \
                    (datesTime['day'] == daily.index[day][2]) 
            dailyData[day,:,:,:] = data[findArr,:,:,:].mean(axis=0)   
    else:
        daily = datesTime.groupby(['year','month','day']).count()
        dailyData = np.empty((daily.shape[0],data.shape[1],data.shape[2]))
        for day in range(0,daily.shape[0]):
            findArr = (datesTime['year'] == daily.index[day][0]) & \
                (datesTime['month'] == daily.index[day][1]) & \
                    (datesTime['day'] == daily.index[day][2]) 
            dailyData[day,:,:] = data[findArr,:,:].mean(axis=0)   
    daily=daily.reset_index()
    
    daily['datetime'] = pd.to_datetime(dict(year=daily['year'], 
                                            month=daily['month'], 
                                            day=daily['day'],
                                            hour=0),
                                       format='%Y-%m-%d %H:00:00').dt.strftime('%Y-%m-%d %H:00:00')

    return dailyData,daily
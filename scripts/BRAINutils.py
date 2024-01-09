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

def createNETCDFtemporalClipper(folderOut,name,data,ds,pollutant,xlon,ylat,datesTime):
    print('===================STARTING netCDFcreator_v1.py=======================')
    dateTimes2 = pd.DataFrame()
    dateTimes2['datetimes'] = datesTime
    dateTimes2['TFLAG']=0
    for ii in range(0,data.shape[0]):
        dateTimes2['TFLAG'][ii] = np.int32(str(datesTime.year[ii])+\
            str(datesTime.month[ii]).zfill(2)+\
                str(datesTime.day[ii]).zfill(2)+\
                    str(datesTime.hour[ii]).zfill(2))
          
    f2 = nc.Dataset(folderOut+'/'+name,'w') #'w' stands for write 
    for gatr in ds.ncattrs() :
        print(gatr)
        try:
            setattr(f2, gatr, ds.__getattribute__(gatr))
        except:
            print('bad var')
    f2.NVARS= data.shape[1]
    f2.HISTORY =''
    setattr(f2, 'VAR-LIST', pollutant)
    f2.NVARS= 1
    f2.NCOLS = data.shape[3]
    f2.NROWS = data.shape[2]
    f2.NVARS = data.shape[1]
    f2.SDATE = dateTimes2['TFLAG'][0]
    f2.FILEDESC = 'Concentration of ' +pollutant +' created by Leonardo Hoinaski - '+ str(datetime.datetime.now())
    f2.HISTORY = ''
    # # Specifying dimensions
    #tempgrp = f.createGroup('vehicularEmissions_data')
    f2.createDimension('TSTEP', None )
    f2.createDimension('DATE-TIME', 2)
    f2.createDimension('LAY', 1)
    f2.createDimension('VAR', data.shape[1])
    f2.createDimension('ROW', data.shape[2])
    f2.createDimension('COL', data.shape[3])
    # Building variables
    TFLAG = f2.createVariable('TFLAG', 'i4', ('TSTEP'))
    # Passing data into variables
    TFLAG[:] = dateTimes2['TFLAG']
    LON = f2.createVariable('LON', 'f4', ( 'ROW','COL'))
    LAT = f2.createVariable('LAT', 'f4', ( 'ROW','COL'))
    LAT[:,:] =  ylat
    LON[:,:] = xlon
    LON.units = 'degrees '
    LAT.units = 'degrees '
    globals()[pollutant] = f2.createVariable(pollutant, np.float32, ('TSTEP', 'LAY', 'ROW','COL'))
    globals()[pollutant][:,:,:,:] = data[:,:,:,:]
    globals()[pollutant].units = ds[pollutant].units
    f2.close()
    return f2

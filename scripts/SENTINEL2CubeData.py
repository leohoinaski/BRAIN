#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:49:15 2024

@author: leohoinaski
"""
import os
import netCDF4 as nc
import BRAINutils
import datetime
import pandas as pd


coarseDomain = 'SENTINEL'
refinedDomain = 'BRAIN' 


NO2 = {
  "Pollutant": "$NO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1880,
  "tag":'NO2',
}

CO = {
  "Pollutant": "CO",
  "Unit": 'ppb',
  "conv": 1000, # Conversão de ppm para ppb
  "tag":'CO',
}

O3 = {
  "Pollutant": "$O_{3}$",
  "Unit": 'ppm',
  "conv": 1,
  "tag":'O3'
}

SO2 = {
  "Pollutant": "$SO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 2620,
  "tag":'SO2'
}

PM10 = {
  "Pollutant": "$PM_{10}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM10',
}

PM25 = {
  "Pollutant": "$PM_{2.5}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM25',
}

pollutants=[NO2]
tinit = datetime.datetime(2010, 1, 1, 0, 0)
time0 = datetime.datetime(1, 1, 1, 0, 0)

for pol in pollutants:
    
    BASE = os.getcwd()
    dataFolder = os.path.dirname(BASE)+'/data'
    coarseDomainPath =  dataFolder+'/' + coarseDomain
    refinedDomain =  dataFolder+'/' + refinedDomain
    year = 2019
    
    # ========BRAIN files============
    os.chdir(refinedDomain)
    print(pol)
    print('Openning netCDF files')
    # Opening netCDF files
    fileType='BRAIN_BASECONC_BR_'+pol['tag']+'_'+str(year)
    prefixed = sorted([filename for filename in os.listdir(refinedDomain) if filename.startswith(fileType)])
    ds = nc.MFDataset(prefixed)
    # Selecting variable
    dataBRAIN = ds[pol['tag']][:]
    # Get datesTime and removing duplicates
    datesTimeBRAIN, dataBRAIN = BRAINutils.fixTimeBRAIN(ds,dataBRAIN)
    latBRAIN = ds['LAT'][:]
    lonBRAIN = ds['LON'][:]
    latBRAINflat = latBRAIN.flatten()
    lonBRAINflat = lonBRAIN.flatten()
    
    os.chdir(coarseDomainPath+'/'+pol['tag'])
    print('Openning netCDF files')
    # Opening netCDF files
    fileType='S5P_OFFL_L2__'+pol['tag']
    prefixed = sorted([filename for filename in os.listdir(coarseDomainPath+'/'+pol['tag']) if filename.startswith(fileType)])
    dfSentinel = pd.DataFrame()
    sentinelAllDf=[]
    times=[]
    ii=0
    for pr in prefixed:
        ds = nc.Dataset(pr)
        try:
            #print (ds.groups['PRODUCT'].variables.keys())
            #print (ds.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'])
            lons = ds.groups['PRODUCT'].variables['longitude'][:][0,:,:].data.flatten()
            lats = ds.groups['PRODUCT'].variables['latitude'][:][0,:,:].data.flatten()
            time = ds.groups['PRODUCT'].variables['time'][:]
            time = datetime.datetime.fromtimestamp(tinit.timestamp()+time[0]).strftime('%Y-%m-%d %H:00:00')
            dataSentinel = ds.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'][0,:,:].data.flatten()  
            
            
            if time0==time:
                if ii == 0:
                    dfSentinel = pd.DataFrame()
                else:
                    ii=ii+1
                print('Same timestamp')
                dfSentinelNew = pd.DataFrame()
                dfSentinelNew['LON'] = lons
                dfSentinelNew['LAT'] = lats
                dfSentinelNew['dataSentinel'] = dataSentinel
                dfSentinel = pd.concat([dfSentinel, dfSentinelNew], ignore_index=True)
            
            else:
                dfSentinel['LON'] = lons
                dfSentinel['LAT'] = lats
                dfSentinel['dataSentinel'] = dataSentinel
                ii=0
                #dfSentinel = pd.DataFrame()
            
            sentinelAllDf.append(dfSentinel.dropna().groupby(by=['LON', 'LAT']).mean())
            print('-----FILE OK!!-----')
            print(time)
        except:
            print(pr)
            print('file without data')
            dfSentinel = pd.DataFrame()
        time0=time
        times.append(time)
        
        
        # plt.figure()
        
        
        
        
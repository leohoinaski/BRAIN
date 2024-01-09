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
import numpy as np

coarseDomain = 'SENTINEL'
refinedDomain = 'BRAIN'


NO2 = {
    "Pollutant": "$NO_{2}$",
    "Unit": '$\u03BCg.m^{-3}$',
    "conv": 1880,
    "tag": 'NO2',
}

CO = {
    "Pollutant": "CO",
    "Unit": 'ppb',
    "conv": 1000,  # Conversão de ppm para ppb
    "tag": 'CO',
}

O3 = {
    "Pollutant": "$O_{3}$",
    "Unit": 'ppm',
    "conv": 1,
    "tag": 'O3'
}

SO2 = {
    "Pollutant": "$SO_{2}$",
    "Unit": '$\u03BCg.m^{-3}$',
    "conv": 2620,
    "tag": 'SO2'
}

PM10 = {
    "Pollutant": "$PM_{10}$",
    "Unit": '$\u03BCg.m^{-3}$',
    "conv": 1,
    "tag": 'PM10',
}

PM25 = {
    "Pollutant": "$PM_{2.5}$",
    "Unit": '$\u03BCg.m^{-3}$',
    "conv": 1,
    "tag": 'PM25',
}

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

pollutants = [NO2]
tinit = datetime.datetime(2010, 1, 1, 0, 0)
time0 = datetime.datetime(1, 1, 1, 0, 0)

for pol in pollutants:

    BASE = os.getcwd()
    dataFolder = os.path.dirname(BASE)+'/data'
    coarseDomainPath = dataFolder+'/' + coarseDomain
    refinedDomain = dataFolder+'/' + refinedDomain
    year = 2019

    # ========BRAIN files============
    os.chdir(refinedDomain)
    print(pol)
    print('Openning netCDF files')
    # Opening netCDF files
    fileType = 'BRAIN_BASECONC_BR_'+pol['tag']+'_'+str(year)
    prefixed = sorted([filename for filename in os.listdir(
        refinedDomain) if filename.startswith(fileType)])
    ds = nc.MFDataset(prefixed)
    # Selecting variable
    dataBRAIN = ds[pol['tag']][:]
    # Get datesTime and removing duplicates
    datesTimeBRAIN, dataBRAIN = BRAINutils.fixTimeBRAIN(ds, dataBRAIN)
    latBRAIN = ds['LAT'][:]
    lonBRAIN = ds['LON'][:]
    latBRAINflat = latBRAIN.flatten()
    lonBRAINflat = lonBRAIN.flatten()
    
    dailyData,daily=dailyAverage(datesTimeBRAIN,dataBRAIN)
    
    matAve=np.empty((dailyData.shape[0],latBRAIN.shape[0],latBRAIN.shape[1]))
    matAve[:,:,:] = np.nan
    
    
    os.chdir(coarseDomainPath+'/'+pol['tag'])
    print('Openning netCDF files')
    # Opening netCDF files
    fileType = 'S5P_OFFL_L2__'+pol['tag']
    prefixed = sorted([filename for filename in os.listdir(
        coarseDomainPath+'/'+pol['tag']) if filename.startswith(fileType)])
    dfSentinel = pd.DataFrame()
    sentinelAllDf = []
    times = []
    ii = 0
    for pr in prefixed:
        ds2 = nc.Dataset(pr)
        
        try:
            time = ds2.groups['PRODUCT'].variables['time'][:]
            time = datetime.datetime.fromtimestamp(
                tinit.timestamp()+time[0]).strftime('%Y-%m-%d 00:00:00')
            print(pr)
            print(time)
            
            if (datesTimeBRAIN.datetime ==time).sum()==1:
                dataInBRAIN = np.empty(lonBRAINflat.shape[0])
                dataInBRAIN[:] = np.nan
                # print (ds2.groups['PRODUCT'].variables.keys())
                # print (ds2.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'])
                lonsOriginal = ds2.groups['PRODUCT'].variables['longitude'][:][0,:, :].data.flatten()
                lonsOriginal[(lonsOriginal>180) | (lonsOriginal<-180)]=np.nan
                lons = lonsOriginal[~np.isnan(lonsOriginal)].copy()
                latsOriginal = ds2.groups['PRODUCT'].variables['latitude'][:][0,:, :].data.flatten()
                latsOriginal[(latsOriginal>90) | (latsOriginal<-90)]=np.nan
                lats = latsOriginal[~np.isnan(latsOriginal)].copy()
    
                dataSentinelOriginal = ds2.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'][0, :,:].data.flatten()
                dataSentinel = dataSentinelOriginal[~np.isnan(lonsOriginal)]
                lats = lats[dataSentinel!=9.96921e+36]
                lons = lons[dataSentinel!=9.96921e+36]
                dataSentinel = dataSentinel[dataSentinel!=9.96921e+36]
                dfSentinel = pd.DataFrame()
                dfSentinel['LON'] = lons
                dfSentinel['LAT'] = lats
                dfSentinel['dataSentinel'] = dataSentinel
                dfSentinel['pixInBRAIN'] = np.nan
                sentinelAllDf.append(
                    dfSentinel.dropna().groupby(by=['LON', 'LAT']).mean().reset_index())
                print('-----FILE OK!!-----')
                #print(time)
                times.append(time)
                pixSentinelInBRAIN=[]
                
                for ii,ll in enumerate(latBRAINflat):
                    #print(str(ii) + ' of '+ str(latBRAINflat.shape[0]))
                    if (latBRAINflat[ii]>np.max(lats)) and (latBRAINflat[ii]<np.max(lats)) and \
                        (lonBRAINflat[ii]>np.max(lons)) and (lonBRAINflat[ii]<np.max(lons)):
                        print('Pixel outside domain') 
                    else:
                        dist = np.sqrt((latBRAINflat[ii]-lats)**2+(lonBRAINflat[ii]-lons)**2)
                        dataInBRAIN[ii] = np.nanmean([dataInBRAIN[ii],dataSentinel[np.argmin(abs(dist))]])

                        
                matAve[daily.datetime ==time,:,:] = \
                    np.nanmean([matAve[daily.datetime ==time,:,:], 
                               dataInBRAIN.reshape(1,matAve.shape[1],matAve.shape[2])],axis=0)
            
            
        except:
            print(pr)
            print('file without data')
    
    name = 'SENTINEL_'+pol['tag']+'_'+str(datesTimeBRAIN.datetime[0])+'_'+str(datesTimeBRAIN.datetime[-1])
    BRAINutils.createNETCDFtemporalClipper(coarseDomainPath,name,matAve,ds,pol['tag'],lonBRAIN,latBRAIN,datesTimeBRAIN)




# import matplotlib.pyplot as plt        
# import geopandas as gpd
# fig,ax = plt.subplots()
# ax.pcolor(lonBRAIN,latBRAIN,matAve[24,:,:])
# np.nansum(matAve[:,:,:])
# shapeBorder = '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
# borda = gpd.read_file(shapeBorder)
# borda.boundary.plot(ax=ax,edgecolor='black',linewidth=0.3)

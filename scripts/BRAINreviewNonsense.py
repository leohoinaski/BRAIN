#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:20:11 2024

Este script foi desenvolvido para responder os pontos do revisor sem noção.

@author: leohoinaski
"""

# Importando bibliotecas
import numpy as np
import numpy.matlib
import netCDF4 as nc
import os
import BRAINutils
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import ismember
import geopandas as gpd
import matplotlib
from scipy.stats import gaussian_kde
import scipy
from shapely.geometry import Point
import pandas as pd
import datetime
# -------------------------------INPUTS----------------------------------------

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

#%%
def cityTimeSeries(cityDataFrame,cityDataFrame2,IBGE_CODE,cmap,legend,
               xlon,ylat,criteria,BASE,pol,aveTime,aqm):
    import matplotlib.dates as mdates

    #cmap = plt.get_cmap(cmap,5)    
    fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [2, 3],
                                            'wspace':0.4, 'hspace':0.4})
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(14*cm, 5*cm)
    
    ax[0].scatter(cityDataFrame2.mean(axis=1),cityDataFrame.mean(axis=1),
                  s=2,alpha=.5,c='darkturquoise')
    x = np.array([cityDataFrame.mean(axis=1),cityDataFrame2.mean(axis=1)])
    x = x[:,~np.isnan(x).any(axis=0)]
    ###calculate Spearman correlation using new_df
    corr, p_value = scipy.stats.spearmanr(x[0,:],x[1,:])
   
    ###insert text with Spearman correlation
    # ax.annotate('ρ = {:.2f}'.format(corr), 
    #         xy=(0.70, 0.9), xycoords='axes fraction', 
    #         fontsize=8, ha='left', va='center')

    ax[0].annotate(aqm+'\nρ = {:.2f}'.format(corr),
            xy=(0.1, 0.9), xycoords='axes fraction', 
            fontsize=8, ha='left', va='center')
    
    y,preds = cityDataFrame2.mean(axis=1),cityDataFrame.mean(axis=1)
    
    ax[0].set_xlim([np.nanmin([y]),np.nanmax([y])])
    ax[0].set_ylim([np.nanmin([preds]),np.nanmax([preds])])
    ax[0].set_xlabel('SENTINEL/TROPOMI\n'+pol['tag']+' tropospheric column \n ($10^{-4}  mol.m^{-2}$)',fontsize=6)
    ax[0].set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
    ax[0].xaxis.set_tick_params(labelsize=6)
    ax[0].yaxis.set_tick_params(labelsize=6)
        
    #ax[0].set_yscale('log')
    #ax[0].set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_xscale('log')

    
    xdata = [datetime.datetime.strptime(x, '%Y-%m-%d 00:00:00') for x in cityDataFrame.mean(axis=1).dropna().index]
    ax[1].fill_between(xdata,cityDataFrame.max(axis=1).dropna(), cityDataFrame.min(axis=1).dropna(),
                     color=cmap(0.8),       # The outline color
                     facecolor=cmap(0.8),
                     edgecolor=None,
                     alpha=0.2,label='Min-Max')          # Transparency of the fill
    ax[1].plot(xdata,cityDataFrame.mean(axis=1).dropna(),
               color=cmap(0.8),linewidth=1,label='BRAIN Average')
    ax[1].xaxis.set_tick_params(labelsize=6)
    ax[1].yaxis.set_tick_params(labelsize=6)
    ax[1].set_ylim([np.nanmin([preds]),np.nanmax([preds])])
    #ax[1].set_ylim([np.nanmin(cityDataFrame)*0.95,np.nanmax(cityDataFrame)*1.05])
    ax[1].set_xlim([np.min(xdata),np.max(xdata)])
    ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax[1].set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)

    # set formatter
    if criteria!=None:
        ax[1].axhline(y=criteria, color='gray', linestyle='--',linewidth=0.5,
                      label='Air quality standard')
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    for label in ax[1].get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    ax[1].legend(prop={'size': 6},loc='upper left', frameon=False)
    
    ax2 = ax[1].twinx()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue","royalblue"])
    xdata = [datetime.datetime.strptime(x, '%Y-%m-%d 00:00:00') for x in cityDataFrame2.mean(axis=1).dropna().index]
    ax2.fill_between(xdata,cityDataFrame2.max(axis=1).dropna(), cityDataFrame2.min(axis=1).dropna(),
                     color=cmap(0.8),       # The outline color
                     facecolor=cmap(0.8),
                     edgecolor=None,
                     alpha=0.2,label='Min-Max')          # Transparency of the fill
    ax2.plot(xdata,cityDataFrame2.mean(axis=1).dropna(),
               color=cmap(0.8),linewidth=1,label='SENTINEL Average')
    ax2.xaxis.set_tick_params(labelsize=6)
    ax2.yaxis.set_tick_params(labelsize=6)
    ax2.set_xlim([np.nanmin([y]),np.nanmax([y])])
    #ax2.set_ylim([np.nanmin(cityDataFrame2)*0.95,np.nanmax(cityDataFrame2)*1.05])
    ax2.set_xlim([np.min(xdata),np.max(xdata)])
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # set formatter
    if criteria!=None:
        ax2.axhline(y=criteria, color='gray', linestyle='--',linewidth=0.5,
                      label='Air quality standard')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    for label in ax[1].get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    ax2.legend(prop={'size': 6},loc='upper right', frameon=False)
    ax2.set_ylabel('SENTINEL/TROPOMI\n'+pol['tag']+' tropospheric column \n ($10^{-4}  mol.m^{-2}$)',fontsize=6)

    
    #ax[1].set_ylabel(cityArea['NM_MUN'].to_string(index=False)+'\n'+legend,fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/timeSeriesNonsense_'+pol['tag']+'.png', format="png",
               bbox_inches='tight',dpi=300)
    return matData.shape
#%%
pollutants=[NO2,CO]

#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
dataFolder = os.path.dirname(BASE)+'/data'
brainFolder =  dataFolder+'/BRAIN'
sentinelFolder = dataFolder+'/SENTINEL' 
year = 2019

shape_path= '/media/leohoinaski/HDD/BRAIN/data/AERONET/aeronet_points/aeronet_points.shp'
aqs = gpd.read_file(shape_path)
aqs.crs = "EPSG:4326"


print('Looping for each variable')
for kk,pol in enumerate(pollutants):
    # ========BRAIN files============
    os.chdir(brainFolder)
    print(pol)
    print('Openning netCDF files')
    # Opening netCDF files
    fileType='BRAIN_BASECONC_BR_'+pol['tag']+'_'+str(year)
    prefixed = sorted([filename for filename in os.listdir(brainFolder) if filename.startswith(fileType)])
    ds = nc.MFDataset(prefixed)
    # Selecting variable
    dataBRAIN = ds[pol['tag']][:]
    # Get datesTime and removing duplicates
    datesTimeBRAIN, dataBRAIN = BRAINutils.fixTimeBRAIN(ds,dataBRAIN)
    latBRAIN = ds['LAT'][:]
    lonBRAIN = ds['LON'][:]
    latBRAINflat = latBRAIN.flatten()
    lonBRAINflat = lonBRAIN.flatten()
    
    dailyData,daily=BRAINutils.dailyAverage(datesTimeBRAIN,dataBRAIN)
    
    # ========SENTINEL files============
    os.chdir(sentinelFolder)
    print(pol)
    print('Openning netCDF files')
    # Opening netCDF files
    fileType='SENTINEL_'+pol['tag']+'_'+str(year)
    prefixed = sorted([filename for filename in os.listdir(sentinelFolder) if filename.startswith(fileType)])
    ds = nc.Dataset(prefixed[0])
    # Selecting variable
    dataSENTINEL = ds[pol['tag']][:]
    # Get datesTime and removing duplicates
    



    #cmap = 'YlOrRd'
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["azure","turquoise"])
    
    legend = 'SENTINEL/TROPOMI \n' +pol['Pollutant'] +' tropospheric column \n ($10^{-4}  mol.m^{-2}$)'
    #legend ='BRAIN'
    for aqm in aqs.Estacao1:
        s,cityMat,cityBuffer=BRAINutils.citiesBufferINdomain(lonBRAIN,latBRAIN,aqs,aqm,'Estacao1')
        #IBGE_CODE=1100205 #    
        cityData,cityDataPoints,cityDataFrame,matData= BRAINutils.dataINcity(dailyData,daily,cityMat,s,aqm)
        cityData2,cityDataPoints2,cityDataFrame2,matData2= BRAINutils.dataINcity(dataSENTINEL,daily,cityMat,s,aqm)
        cityTimeSeries(cityDataFrame,(10**4)*cityDataFrame2,aqm,cmap,legend,
                       lonBRAIN,latBRAIN,None,BASE,pol,24,aqm)
        
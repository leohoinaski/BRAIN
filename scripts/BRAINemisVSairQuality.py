#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:09:12 2024

Este script é utilizado para analisar a relação entre emissão local e qualidade
do ar no mesmo pixel. 



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
# -------------------------------INPUTS----------------------------------------


NO2 = {
  "Pollutant": "$NO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1880,
  "tag":'NO2',
  "Criteria": 200,
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

emisTypes = ['BRAVES','FINN','IND2CMAQ','MEGAN']

#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
dataFolder = os.path.dirname(BASE)+'/data'
airQualityFolder =  dataFolder+'/BRAIN'
emissFolder =  dataFolder+'/EMIS'
domain = 'BR'
year = '2020'

print('Looping for each variable')
for kk,pol in enumerate(pollutants):
    
    # ======== EMIS files============
    # Selecting variable
    if pol['tag']=='CO':
        polEmis = 'CO'
    elif pol['tag']=='O3':
        polEmis = 'VOC'
    elif pol['tag']=='SO2':
        polEmis = 'SOX'
    elif pol['tag'] == 'PM25':
        polEmis = 'PM10'
    elif pol['tag'] == 'NO2':
        polEmis = 'NOX'
    elif pol['tag'] == 'PM10':
        polEmis = 'PM10'
        
    os.chdir(emissFolder)
    print('Openning netCDF files')
    # Opening netCDF files
    
    for ii, emisType in enumerate(emisTypes):
        fileType='BRAIN_BASEMIS_'+domain+'_2019_'+emisType+'_'+polEmis+'_'+str(year)
        prefixed = sorted([filename for filename in os.listdir(emissFolder) if filename.startswith(fileType)])
        ds = nc.Dataset(prefixed[0])
        if ii==0:
            dataEMIS = ds[polEmis][0:8759,:,:,:]
        else:
            dataEMIS = dataEMIS+ds[polEmis][0:8759,:,:,:]
            
    os.chdir(os.path.dirname(BASE))
    datesTimeEMIS, dataEMIS = BRAINutils.fixTimeBRAINemis(ds,dataEMIS[0:8759,:,:,:])
    
    # ========BRAIN files============
    os.chdir(airQualityFolder)
    print(pol)
    print('Openning netCDF files')
    # Opening netCDF files
    fileType='BRAIN_BASECONC_'+domain+'_'+pol['tag']+'_'+str(year)
    prefixed = sorted([filename for filename in os.listdir(airQualityFolder) if filename.startswith(fileType)])
    ds = nc.Dataset(prefixed[0])
    # Selecting variable
    dataBRAIN = ds[pol['tag']][:]
    # Get datesTime and removing duplicates
    datesTimeBRAIN, dataBRAIN = BRAINutils.fixTimeBRAIN(ds,dataBRAIN)
    latBRAIN = ds['LAT'][:]
    lonBRAIN = ds['LON'][:]
    latBRAINflat = latBRAIN.flatten()
    lonBRAINflat = lonBRAIN.flatten()
    os.chdir(os.path.dirname(BASE))
    
    
fig, ax = plt.subplots()
ax.scatter(dataBRAIN[24:1000,:,:,:].flatten()*pol['conv'],
           dataEMIS[24:1000,:,:,:].flatten(),
           s=1,alpha=.2,c='red')
if pol['Criteria']!=None:
    ax.axhline(y=pol['Criteria'], color='gray', linestyle='--',linewidth=1,
                  label='Air quality standard')
    ax.axvline(x=np.percentile(dataEMIS[24:1000,:,:,:].flatten()[dataBRAIN[24:1000,:,:,:].flatten()*pol['conv']>pol['Criteria']],50), 
               color='gray', linestyle='--',linewidth=1,
                   label='Lowest significant emission')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('Air quality\n'+pol['tag']+ ' ('+pol['Unit']+')',fontsize=8)
ax.set_xlabel('Emission\n'+polEmis ,fontsize=8)

test =dataEMIS[24:1000,:,:,:].flatten()[dataBRAIN[24:1000,:,:,:].flatten()*pol['conv']>pol['Criteria']]

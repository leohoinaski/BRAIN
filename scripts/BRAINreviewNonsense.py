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

pollutants=[NO2]

#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
dataFolder = os.path.dirname(BASE)+'/data'
brainFolder =  dataFolder+'/BRAIN'
sentinelFolder = dataFolder+'/SENTINEL' 
year = 2019

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
    

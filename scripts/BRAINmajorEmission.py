#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:26:01 2024

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
import matplotlib as mpl

# -------------------------------INPUTS----------------------------------------


NO2 = {
  "Pollutant": "$NO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1880,
  "tag":'NO2',
  #"Criteria": 260, # 260, 240, 220, 200
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

pollutants=[NO2,SO2,O3,PM10,PM25]
emisTypes = ['BRAVES','FINN','IND2CMAQ','MEGAN']

#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
rootFolder = os.path.dirname(os.path.dirname(BASE))
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
    emisAve = []
    emisMax = []
    for ii, emisType in enumerate(emisTypes):
        fileType='BRAIN_BASEMIS_'+domain+'_2019_'+emisType+'_'+polEmis+'_'+str(year)
        prefixed = sorted([filename for filename in os.listdir(emissFolder) if filename.startswith(fileType)])
        ds = nc.Dataset(prefixed[0])
        dataEMIS = np.nanpercentile(ds[polEmis][0:8759,:,:,:],50,axis=0)
        emisAve.append(dataEMIS)
        emisMax(np.nanpercentile(ds[polEmis][0:8759,:,:,:],99,axis=0))

    
    latBRAIN = ds['LAT'][:]
    lonBRAIN = ds['LON'][:]
    
    os.chdir(os.path.dirname(BASE))
    datesTimeEMIS, dataEMIS = BRAINutils.fixTimeBRAINemis(ds,dataEMIS[0:8759,:,:,:])
    
 
    #% Removendo dados fora do Brasil
    
    shape_path= rootFolder+'/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
    dataShp = gpd.read_file(shape_path)
    
    def dataINshape(xlon,ylat,uf):
        s = gpd.GeoSeries(map(Point, zip(xlon.flatten(), ylat.flatten())))
        s = gpd.GeoDataFrame(geometry=s)
        s.crs = "EPSG:4326"
        s.to_crs("EPSG:4326")
        uf.crs = "EPSG:4326"
        pointIn = uf['geometry'].buffer(0.1).clip(s).explode()
        pointIn = gpd.GeoDataFrame({'geometry':pointIn}).reset_index()
        lia, loc = ismember.ismember(np.array((s.geometry.x,s.geometry.y)).transpose(),
                            np.array((pointIn.geometry.x,pointIn.geometry.y)).transpose(),'rows')
        s['mask']=0
        s['mask'][lia]=1
        cityMat = np.reshape(np.array(s['mask']),(xlon.shape[0],xlon.shape[1]))
        return s,cityMat
    
    uf = dataShp[dataShp['NM_PAIS']=='Brasil']
    s,cityMat=dataINshape(lonBRAIN,latBRAIN,uf)
    
    print('Removing emissions outsitde domain')
    for emi in emisMax:
        emi[:,:,cityMat==0] = np.nan
        emi[:,:,cityMat==0] = np.nan
    for emi in emisAve:
        emi[:,:,cityMat==0] = np.nan
        emi[:,:,cityMat==0] = np.nan
        
        
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:07:40 2024

@author: Leonardo.Hoinaski
"""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import netCDF4 as nc
from shapely.geometry import Point
import ismember
import os

# -------------------------------INPUTS----------------------------------------


NO2 = {
  "Pollutant": "$NO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1880,
  "tag":'NO2',
  #"Criteria": 260, # 260, 240, 220, 200
  "Criteria_ave": 1,
  #"criterias" : [260,240,220,200],
  "criterias" : [200],
  "Criteria_average": '1-h average',
}

CO = {
  "Pollutant": "CO",
  "Unit": 'ppb',
  "conv": 1000, # Conversão de ppm para ppb
  "tag":'CO',
  "Criteria_ave": 8,
  "criterias" : [9000],
  "Criteria_average": '8-h average',
}

O3 = {
  "Pollutant": "$O_{3}$",
  "Unit": 'ppm',
  "conv":1962 ,
  "tag":'O3',
  "Criteria_ave": 8,
  #"criterias" : [140,130,120,100],
  "criterias" : [100],
  "Criteria_average": '8-h average',
}

SO2 = {
  "Pollutant": "$SO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 2620,
  "tag":'SO2',
  "Criteria_ave": 24,
  #"criterias" : [125,50,40,30,20],
  "criterias" : [40],
  "Criteria_average": '24-h average',
  
}

PM10 = {
  "Pollutant": "$PM_{10}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM10',
  "Criteria_ave": 24,
  #"criterias" : [120,100,75,50,45],
  "criterias" : [45],
  "Criteria_average": '24-h average',
}

PM25 = {
  "Pollutant": "$PM_{2.5}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM25',
  "Criteria_ave": 24,
  #"criterias" : [60,50,37,25,15],
  "criterias" : [15],
  "Criteria_average": '24-h average',
}

pollutants=[NO2,SO2,O3,PM10,PM25]
BASE = os.getcwd()
rootFolder = os.path.dirname(os.path.dirname(BASE))
Q4path = os.path.dirname(BASE)+'/data/Q4'
tablepath = os.path.dirname(BASE)+'/tables'
gridPath =os.path.dirname(BASE)+'/data/BRAIN/BRAIN_BASECONC_SC_PM10_2019_01_01_00_to_2019_12_31_00.nc'
shape = os.path.dirname(os.path.dirname(BASE))+'/shapefiles/SC_Municipios_2022/SC_Municipios_2022.shp'
emisTypes = ['BRAVES','FINN','IND2CMAQ','MEGAN']
emisNames = ['Vehicular', 'Fire', 'Industrial', 'Biogenic']
colors =['#0c8b96','#f51b1b','#fcf803','#98e3ad']

def dataINshape(xlon,ylat,uf):
    #print(uf)
    s = gpd.GeoSeries(map(Point, zip(xlon.flatten(), ylat.flatten())))
    s = gpd.GeoDataFrame(geometry=s)
    s.crs = "EPSG:4326"
    s.to_crs("EPSG:4326")
    
    uf = gpd.GeoDataFrame(geometry=[uf.geometry.buffer(0.1)])
    pointIn = uf.clip(s).explode().reset_index()

    pointIn = gpd.GeoDataFrame(geometry = pointIn.iloc[:,-1])
    lia, loc = ismember.ismember(np.array((s.geometry.x,s.geometry.y)).transpose(),
                        np.array((pointIn.geometry.x,pointIn.geometry.y)).transpose(),'rows')
    s['mask']=0
    s['mask'][lia]=1
    cityMat = np.reshape(np.array(s['mask']),(xlon.shape[0],xlon.shape[1]))
    return s,cityMat

for pol in pollutants:
    for jj,pp in enumerate(pol['criterias']):
        path = Q4path+'/Q4_'+pol['tag']+'_SC_'+str(pol['criterias'][jj])+'.npy'
        with open(path, 'rb') as f:
            a = np.load(f)
            b = np.load(f)
            c = np.load(f)     
        path = tablepath+'/MajorAveEmitter_'+pol['tag']+'_SC.npy'
        with open(path, 'rb') as f:
            majorAve = np.load(f)
        path = tablepath+'/MajorMaxEmitter_'+pol['tag']+'_SC.npy'
        with open(path, 'rb') as f:
            majorMax = np.load(f)
                
        geoData = gpd.read_file(shape)
        grid = nc.Dataset(gridPath)
        lat = grid['LAT'][:]
        lon = grid['LON'][:]
        
        geoData['CriticalPixels']=np.nan
        geoData['CriticalEvents']=np.nan
        geoData['MeanEf']=np.nan
        geoData['MinEf']=np.nan
        geoData['MaxEf']=np.nan
        geoData['MeanEmis']=np.nan
        geoData['MinEmis']=np.nan
        geoData['MaxEmis']=np.nan
        geoData['MajorAveVehicular']=np.nan
        geoData['MajorAveFire']=np.nan
        geoData['MajorAveIndustrial']=np.nan
        geoData['MajorAveBiogenic']=np.nan
        geoData['MajorMaxVehicular']=np.nan
        geoData['MajorMaxFire']=np.nan
        geoData['MajorMaxIndustrial']=np.nan
        geoData['MajorMaxBiogenic']=np.nan
        for ii, uf in  geoData.iterrows():
            print(ii)
            s,cityMat = dataINshape(lon,lat,uf)
            critical = a[cityMat==1]
            geoData['CriticalPixels'][ii] = np.nansum(critical)
            critical = c[:,0,cityMat==1]
            geoData['CriticalEvents'][ii] = np.nansum(critical>0)
            geoData['MeanEf'][ii] = np.nanmean(critical)
            geoData['MinEf'][ii] = np.nanmin(critical)
            geoData['MaxEf'][ii] = np.nanmax(critical)
            critical = b[:,0,cityMat==1]
            geoData['MeanEmis'][ii] = np.nanmean(critical)
            geoData['MinEmis'][ii] = np.nanmin(critical)
            geoData['MaxEmis'][ii] = np.nanmax(critical)      
            critical = majorAve[cityMat==1]
            geoData['MajorAveVehicular'][ii] = np.sum(critical==0)
            geoData['MajorAveFire'][ii] = np.sum(critical==1)
            geoData['MajorAveIndustrial'][ii] = np.sum(critical==2)
            geoData['MajorAveBiogenic'][ii] = np.sum(critical==3)
            critical = majorMax[cityMat==1]
            geoData['MajorMaxVehicular'][ii] = np.sum(critical==0)
            geoData['MajorMaxFire'][ii] = np.sum(critical==1)
            geoData['MajorMaxIndustrial'][ii] = np.sum(critical==2)
            geoData['MajorMaxBiogenic'][ii] = np.sum(critical==3)
        geoData.drop('geometry',axis=1).to_csv(tablepath+'/crititalCity_'+
                                               pol['tag']+'_'+
                                               str(pol['criterias'][jj])+'.csv') 
        # fig ,ax = plt.subplots()
        # ax.pcolor(lon,lat,b[0,0,:,:])
        # geoData.boundary.plot(ax=ax)
        
        # fig ,ax = plt.subplots()
        # geoData.plot(ax=ax,column='CriticalEvents')

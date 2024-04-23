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

path =r"C:\Users\Leonardo.Hoinaski\Downloads\Q4_NO2_SC_260.npy"

gridPath =r"C:\Users\Leonardo.Hoinaski\Downloads\BRAIN_BackgroundCONC_SO2_AnnualAverage_2019(2).nc"

shape = r"C:\Users\Leonardo.Hoinaski\Documents\SC_Municipios_2022\SC_Municipios_2022.shp"

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


with open(path, 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    c = np.load(f)



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
for ii, uf in  geoData.iterrows():
    print(ii)
    s,cityMat = dataINshape(lon,lat,uf)
    critical = a[cityMat==1]
    geoData['CriticalPixels'][ii] = np.nansum(critical)
    critical = c[:,0,cityMat==1]
    geoData['CriticalEvents'][ii] = np.nansum(c>0)
    geoData['MeanEf'][ii] = np.nanmean(c)
    geoData['MinEf'][ii] = np.nanmin(c)
    geoData['MaxEf'][ii] = np.nanmax(c)
    critical = b[:,0,cityMat==1]
    geoData['MeanEmis'][ii] = np.nanmean(b)
    geoData['MinEmis'][ii] = np.nanmin(b)
    geoData['MaxEmis'][ii] = np.nanmax(b)
    

fig ,ax = plt.subplots()
ax.pcolor(lon,lat,b[0,0,:,:])
geoData.boundary.plot(ax=ax)

fig ,ax = plt.subplots()
geoData.plot(ax=ax,column='CriticalEvents')

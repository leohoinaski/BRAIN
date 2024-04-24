#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:21:34 2023

@author: leohoinaski
"""

import os
import numpy as np
import geopandas as gpd
#from datetime import datetime
import netCDF4 as nc
#import pandas as pd
import temporalStatistics as tst
import GAr_figs as garfig
import matplotlib
import shutil
import BRAINutils
import ismember
from shapely.geometry import Point

#%INPUTS
year = 2020
GDNAM = 'SC_2019'
fileTypes = ['BRAVES','FINN','IND2CMAQ','MEGAN']

emissType=['Vehicular', 'Fire', 'Industrial', 'Biogenic']

BASE = os.getcwd()
rootFolder = os.path.dirname(os.path.dirname(BASE))

emisDataFolder = os.path.dirname(BASE)+'/data/EMIS/'

borderShape = rootFolder+'/shapefiles/Brasil.shp'
borderShape2= rootFolder+'/shapefiles/SC_Mesorregioes_2022/SC_Mesorregioes_2022.shp'
cityShape = rootFolder+'/shapefiles/BR_Municipios_2020.shp'


colors =['#0c8b96','#f51b1b','#fcf803','#98e3ad']
colors2 = ["crimson","gold","red","lightgreen"]

cmap = [
        matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","beige","crimson","purple"]),
        matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","yellow","gold",'#f51b1b']),
        matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","gray","red","purple"]),
        matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","beige","lightgreen","darkgreen"])]

# Trim domain
left = 40
right = 20
top=95
bottom=20


NO2 = {
  "Pollutant": "$NO_{2}$",
  "Unit": '$mol.s^{-1}$',
  "conv": 1880,
  "tag":'NOX',
  #"Criteria": 260, # 260, 240, 220, 200
}

CO = {
  "Pollutant": "CO",
  "Unit": '$mol.s^{-1}$',
  "conv": 1000, # Conversão de ppm para ppb
  "tag":'CO',
}

O3 = {
  "Pollutant": "$O_{3}$",
  "Unit": '$mol.s^{-1}$',
  "conv": 1,
  "tag":'O3'
}

SO2 = {
  "Pollutant": "$SO_{2}$",
  "Unit": '$mol.s^{-1}$',
  "conv": 2620,
  "tag":'SOX'
}

PM10 = {
  "Pollutant": "$PM_{10}$",
  "Unit": '$g.s^{-1}$',
  "conv": 1,
  "tag":'PM10',
}

PM25 = {
  "Pollutant": "$PM_{2.5}$",
  "Unit": '$g.s^{-1}$',
  "conv": 1,
  "tag":'PM25',
}

pollutants=[CO,NO2,SO2,O3,PM10,PM25]


def dataINshape(xlon,ylat,uf):
    s = gpd.GeoSeries(map(Point, zip(xlon.flatten(), ylat.flatten())))
    s = gpd.GeoDataFrame(geometry=s)
    s.crs = "EPSG:4326"
    s.to_crs("EPSG:4326")
    uf.crs = "EPSG:4326"
    pointIn = uf['geometry'].clip(s).explode()
    pointIn = gpd.GeoDataFrame({'geometry':pointIn}).reset_index()
    lia, loc = ismember.ismember(np.array((s.geometry.x,s.geometry.y)).transpose(),
                        np.array((pointIn.geometry.x,pointIn.geometry.y)).transpose(),'rows')
    s['mask']=0
    s['mask'][lia]=1
    cityMat = np.reshape(np.array(s['mask']),(xlon.shape[0],xlon.shape[1]))
    return s,cityMat

            
#%% ------------------------------PROCESSING-----------------------------------
print('--------------Start GAr_emissAnalysis.py------------')
print('creating folders')
os.makedirs(emisDataFolder+'/EMISfigures', exist_ok=True)
os.makedirs(emisDataFolder+'/EMIStables', exist_ok=True)



#Looping each fileTypes
baseFile = 'BRAIN_BASEMIS_'+GDNAM+'_'
for count, fileType in enumerate(fileTypes):
    print(fileType)
    # Moving to dir
    os.chdir(emisDataFolder)
    #Looping each pollutant
    for pol in pollutants:
        print(pol['tag'])
        # Selecting files and variables
        prefixed = sorted([filename for filename in os.listdir(emisDataFolder)  if filename.startswith(baseFile+fileType+'_'+pol['tag']+'_'+str(year))])
        print(prefixed)
        os.chdir(BASE)
        # Opening netCDF files
        if len(prefixed)>0:
            ds = nc.Dataset(emisDataFolder+'/'+prefixed[0])
            # Selecting variable
            dataOriginal = ds[pol['tag']][:]
            datesTime, data = BRAINutils.fixTimeBRAIN(ds,dataOriginal)
            data = np.nansum(data,axis=0)[0,:,:]
            xlon = ds['LON'][:]
            ylat = ds['LAT'][:]
            
            dataT,xvT,yvT= tst.trimBorders(data,xlon,ylat,left,right,top,bottom)
            
            # Analyzing by city
            cities = gpd.read_file(cityShape)
            cities.crs = "EPSG:4326"
            cities = cities[cities['SIGLA_UF']=='SC']
            s0,cityMat0 = tst.citiesINdomain(xlon, ylat, cities)
            s,cityMat = tst.citiesINdomain(xvT, yvT, cities)
            matDataAll=dataT.copy()
            matDataAll[np.isnan(cityMat,)]=0
            idxMax = np.unravel_index((matDataAll[:,:]).argmax(),
                          (matDataAll[:,:]).shape)

            # ============================Figures==================================
            
            # Spatial distribution
            legend = 'Annual '+emissType[count]+' emission of ' + pol['Pollutant'] + ' ('+'$mol.year^{-1}$'+')'
            garfig.spatialEmissFig(dataT,xvT,yvT,
                                   legend,cmap[count],borderShape,emisDataFolder+'/EMISfigures',
                                   pol['tag'],emissType[count],borderShape2=borderShape2)
            
            dataShp = gpd.read_file(borderShape2) 
            

            dataUF=[]
            for jj,state in enumerate(dataShp['NM_MESO']):
                #uf = dataShp[dataShp['UF']==state]
                uf = dataShp[dataShp['NM_MESO']==state]
                s,cityMat=dataINshape(xvT,yvT,uf)
                dataUF.append(dataT[cityMat==1])
            
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots()
            cm = 1/2.54  # centimeters in inches
            fig.set_size_inches(12*cm, 6*cm)
            ax.boxplot(dataUF,notch=True, patch_artist=True,
                        boxprops=dict(facecolor=colors2[count], color='black'),
                        #capprops=dict(color='#fcf803'),
                        #whiskerprops=dict(color='#fcf803'),
                        flierprops=dict(color=colors2[count], markeredgecolor='black',markersize= 3),
                        medianprops=dict(color='black'))
            ax.set_yscale('symlog')
            maximum = max(arr.max() for arr in dataUF)
            ax.set_ylim([0 ,maximum*1.1 ])
            ax.set_ylabel(pol['Pollutant'] + '\n'+ emissType[count]+' emission' +  '\n ('+'$mol.year^{-1}$'+')',fontsize=8)
            ax.set_xticklabels(dataShp['NM_MESO'].str.replace(' ','\n'),fontsize=8,rotation=30)
            ax.tick_params(axis='both', which='major', labelsize=8)
            fig.tight_layout()
            fig.savefig(emisDataFolder+'/EMISfigures/boxplotEmissions_'+fileType+'_'+pol['tag']+'.png',
                        dpi=300)
            
    
        # # Saving data for each city
        for IBGE_CODE in cities['CD_MUN']:
            uf = cities[cities['CD_MUN']==IBGE_CODE]
            cityData,cityDataPoints,cityDataFrame,matData = tst.dataINcity(data,datesTime,cityMat0,s0,IBGE_CODE)
            os.makedirs(emisDataFolder+'/EMIStables'+'/'+pol['tag'], exist_ok=True)
            cityDataFrame.to_csv(emisDataFolder+'/EMIStables'+'/'+pol['tag']+'/'+pol['tag']+'_'+str(IBGE_CODE)+'.csv')
        
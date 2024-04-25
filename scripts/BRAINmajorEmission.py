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
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
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
emisNames = ['Vehicular', 'Fire', 'Industrial', 'Biogenic']
colors =['#0c8b96','#f51b1b','#fcf803','#98e3ad']
#pollutants=[CO]
#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
rootFolder = os.path.dirname(os.path.dirname(BASE))
dataFolder = os.path.dirname(BASE)+'/data'
airQualityFolder =  dataFolder+'/BRAIN'
emissFolder =  dataFolder+'/EMIS'
domain = 'SC'
year = '2020'
majorEmitters=[]
majorEmittersAve=[]
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
    emisNamesUpdated =[]
    updatedColors =[]
    for ii, emisType in enumerate(emisTypes):
        fileType='BRAIN_BASEMIS_'+domain+'_2019_'+emisType+'_'+polEmis+'_'+str(year)
        prefixed = sorted([filename for filename in os.listdir(emissFolder) if filename.startswith(fileType)])
        if len(prefixed)>0:
            ds = nc.Dataset(prefixed[0])
            emisAve.append(np.nanmean(ds[polEmis][0:8759,:,:,:],axis=0))
            emisMax.append(np.nanpercentile(ds[polEmis][0:8759,:,:,:],99,axis=0))
            emisNamesUpdated.append(emisNames[ii])
            updatedColors.append(colors[ii])

    dataEMIS = ds[polEmis][0:8759,:,:,:]
    latBRAIN = ds['LAT'][:]
    lonBRAIN = ds['LON'][:]
    
    os.chdir(os.path.dirname(BASE))
    datesTimeEMIS, dataEMIS = BRAINutils.fixTimeBRAINemis(ds,dataEMIS[0:8759,:,:,:])
    
 
    #% Removendo dados fora do Brasil
    
    #shape_path= rootFolder+'/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
    shape_path= rootFolder+'/shapefiles/Brasil.shp'
    dataShp = gpd.read_file(shape_path)

    
    def dataINshape(xlon,ylat,uf):
        s = gpd.GeoSeries(map(Point, zip(xlon.flatten(), ylat.flatten())))
        s = gpd.GeoDataFrame(geometry=s)
        s.crs = "EPSG:4326"
        s.to_crs("EPSG:4326")
        uf.crs = "EPSG:4326"
        pointIn = uf['geometry'].buffer(0.01).clip(s).explode()
        pointIn = gpd.GeoDataFrame({'geometry':pointIn}).reset_index()
        lia, loc = ismember.ismember(np.array((s.geometry.x,s.geometry.y)).transpose(),
                            np.array((pointIn.geometry.x,pointIn.geometry.y)).transpose(),'rows')
        s['mask']=0
        s['mask'][lia]=1
        cityMat = np.reshape(np.array(s['mask']),(xlon.shape[0],xlon.shape[1]))
        return s,cityMat
    
    #uf = dataShp[dataShp['NM_PAIS']=='Brasil']
    uf = dataShp[dataShp['UF']=='SC']
    s,cityMat=dataINshape(lonBRAIN,latBRAIN,uf)
    
    print('Removing emissions outsitde domain')
    for emi in emisMax:
        emi[:,cityMat==0] = np.nan
        emi[:,cityMat==0] = np.nan
    for emi in emisAve:
        emi[:,cityMat==0] = np.nan
        emi[:,cityMat==0] = np.nan
        
    
    majorEmitter = np.argmax(np.vstack((emisMax)),axis=0).astype(float)
    majorEmitter[cityMat==0] = np.nan
    majorEmitter[np.nansum(emisMax,axis=0)[0,:,:]<=0]=np.nan
    majorEmitters.append(majorEmitter)
    with open(os.path.dirname(BASE)+'/tables'+'/MajorMaxEmitter_'+pol['tag']+'_'+domain+'.npy', 'wb') as f:
        np.save(f, majorEmitter)

    
    majorEmitterAve = np.argmax(np.vstack((emisAve)),axis=0).astype(float)
    majorEmitterAve[cityMat==0] = np.nan
    majorEmitterAve[np.nansum(emisAve,axis=0)[0,:,:]<=0]=np.nan
    majorEmittersAve.append(majorEmitterAve)
    with open(os.path.dirname(BASE)+'/tables'+'/MajorAveEmitter_'+pol['tag']+'_'+domain+'.npy', 'wb') as f:
        np.save(f, majorEmitter)

    #%%
    #shape_path= rootFolder+'/shapefiles/BR_regions.shp'
    shape_path= rootFolder+'/shapefiles/Brasil.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
    #dataShp = gpd.read_file(shape_path)
    shape_path= rootFolder+'/shapefiles/SC_Mesorregioes_2022/SC_Mesorregioes_2022.shp'
    dataShp = gpd.read_file(shape_path)
    #dataShp= uf.copy()
    xb = [np.nanmin(lonBRAIN[~np.isnan(majorEmitter)[:,:]]),
          np.nanmax(lonBRAIN[~np.isnan(majorEmitter)[:,:]])]
    yb = [np.nanmin(latBRAIN[~np.isnan(majorEmitter)[:,:]]),
          np.nanmax(latBRAIN[~np.isnan(majorEmitter)[:,:]])]
    
    fig,ax = plt.subplots(2,2,height_ratios=[4, 1])
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(18*cm, 15*cm)
    cmap = matplotlib.colors.ListedColormap(updatedColors)
    bound = np.array(np.array(range(0,len(emisNamesUpdated)+1)))
    heatmap = ax[0,0].pcolor(lonBRAIN,latBRAIN,majorEmitterAve,
                        cmap=cmap)
    # Preparing borders for the legend
    bound_prep = np.round(bound * 7, 2)
    # Creating 8 Patch instances
    ax[0,0].legend([mpatches.Patch(color=cmap(b)) for b in bound[:-1]],
               emisNamesUpdated,loc='lower left' ,fontsize=8, markerscale=15, frameon=False)
    ax[0,0].set_xlim([xb[0], xb[1]])
    ax[0,0].set_ylim([yb[0], yb[1]])
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])    
    dataShp.boundary.plot(ax=ax[0,0],edgecolor='black',linewidth=0.3)
    ax[0,0].set_frame_on(False)
    ax[0,0].text(0.0, 0.45, 'a) Average'+'\n'+polEmis+' major emitters', transform=ax[0,0].transAxes,
            size=7)
    
    heatmap = ax[0,1].pcolor(lonBRAIN,latBRAIN,majorEmitter,
                        cmap=cmap)
    ax[0,1].set_xlim([xb[0], xb[1]])
    ax[0,1].set_ylim([yb[0], yb[1]])
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])    
    dataShp.boundary.plot(ax=ax[0,1],edgecolor='black',linewidth=0.3)
    ax[0,1].set_frame_on(False)
    ax[0,1].text(0.0, 0.45, 'b) 99° percentile'+'\n'+polEmis+' major emitters', transform=ax[0,1].transAxes,
            size=7)
    #fig.tight_layout()

    
    
    # Por estado
    #shape_path= rootFolder+'/shapefiles/Brasil.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
    shape_path= rootFolder+'/shapefiles/SC_Mesorregioes_2022/SC_Mesorregioes_2022.shp'

    dataShp = gpd.read_file(shape_path)
    
    dataBox=[]
    dataBoxAve=[]
    
    dfMajor=pd.DataFrame()
    # dfMajor['state'] = dataShp['UF']
    # dfMajor['region'] = dataShp['REGIAO']
    dfMajor['state'] = dataShp['NM_MESO']
    dfMajor['region'] = dataShp['NM_MESO']
    dfMajorAve=pd.DataFrame()
    #dfMajorAve['state'] = dataShp['UF']
    #dfMajorAve['region'] = dataShp['REGIAO']
    dfMajorAve['state'] = dataShp['NM_MESO']
    dfMajorAve['region'] = dataShp['NM_MESO']
    for kk, source in enumerate(emisNamesUpdated):
        dfMajor[source] = np.nan
        dfMajorAve[source] = np.nan
        
    #for jj,state in enumerate(dataShp['UF']):
    for jj,state in enumerate(dataShp['NM_MESO']):
        #uf = dataShp[dataShp['UF']==state]
        uf = dataShp[dataShp['NM_MESO']==state]
        s,cityMat=dataINshape(lonBRAIN,latBRAIN,uf)
        dataBox.append(majorEmitter[cityMat==1][~np.isnan(majorEmitter[cityMat==1])])
        dataBoxAve.append(majorEmitter[cityMat==1][~np.isnan(majorEmitterAve[cityMat==1])])
        
        for kk, source in enumerate(emisNamesUpdated):
            dfMajor[source][jj] = np.sum(majorEmitter[cityMat==1][~np.isnan(majorEmitter[cityMat==1])]==kk)
            dfMajorAve[source][jj] = np.sum(majorEmitterAve[cityMat==1][~np.isnan(majorEmitterAve[cityMat==1])]==kk)

    dfMajor['total']=dfMajor[emisNamesUpdated].sum(axis=1)
    dfMajor[emisNamesUpdated] = 100*dfMajor[emisNamesUpdated]/np.repeat(dfMajor['total'].values,len(emisNamesUpdated),axis=0).reshape(dfMajor[emisNamesUpdated].shape)
    dfMajor= dfMajor.sort_values(by=['region']).reset_index()
    try:
        dfMajor.drop('level_0', axis=1, inplace=True)
    except:
        print('')
        
    dfMajorAve['total']=dfMajorAve[emisNamesUpdated].sum(axis=1)
    dfMajorAve[emisNamesUpdated] = 100*dfMajorAve[emisNamesUpdated]/np.repeat(dfMajorAve['total'].values,len(emisNamesUpdated),axis=0).reshape(dfMajorAve[emisNamesUpdated].shape)
    dfMajorAve= dfMajorAve.sort_values(by=['region']).reset_index()
    try:
        dfMajorAve.drop('level_0', axis=1, inplace=True)
    except:
        print('')
            
        
    
    dfMajorAve[emisNamesUpdated].plot.bar(stacked=True,color=updatedColors,ax=ax[1,0])
    ax[1,0].yaxis.set_major_formatter(mtick.PercentFormatter())    
    ax[1,0].set_xticks(np.array(dfMajorAve['state'].index), dfMajorAve['state'].str.replace(' ','\n'),fontsize=7)
    ax[1,0].set_ylim([0,100])
    ax[1,0].tick_params(axis='both', which='major', labelsize=6)
    ax[1,0].set_ylabel('c) Average \n'+polEmis+' Major source (%)' ,fontsize=8)
    #cm = 1/2.54  # centimeters in inches
    #fig.set_size_inches(18*cm, 6*cm)
    # fill with colors
    #ax[1,0].legend(loc='lower right' ,fontsize=8, markerscale=15)
    ax[1,0].get_legend().remove()
    
    dfMajor[emisNamesUpdated].plot.bar(stacked=True,color=updatedColors,ax=ax[1,1])
    ax[1,1].yaxis.set_major_formatter(mtick.PercentFormatter())    
    ax[1,1].set_xticks(np.array(dfMajor['state'].index), dfMajor['state'].str.replace(' ','\n'),fontsize=7)
    ax[1,1].set_ylim([0,100])
    ax[1,1].tick_params(axis='both', which='major', labelsize=6)
    ax[1,1].set_ylabel('d) 99° percentile'+'\n'+polEmis+ ' Major source (%)' ,fontsize=8)
    # fill with colors
    #ax[1,1].legend(loc='lower right' ,fontsize=8, markerscale=15)
    ax[1,1].get_legend().remove()
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/majorEmitters_'+domain+'_'+polEmis+'.png', format="png",
              bbox_inches='tight',dpi=300)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:55:09 2024

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
import temporalStatistics as tst
# -------------------------------INPUTS----------------------------------------
coarseDomain = 'MERRA'
refinedDomain = 'BRAIN' 

# Trim domain
left = 40
right = 20
top=95
bottom=20

# left = 0
# right = 0
# top=0
# bottom=0


#%%


NO2 = {
  "Pollutant": "$NO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1880,
  "tag":'NO2',
  #"Criteria": 260, # 260, 240, 220, 200
  "Criteria_ave": 1,
  "criterias" : [260,240,220,200],
  "criteria" : 260,
  "Criteria_average": '1-h average',
}

CO = {
  "Pollutant": "CO",
  "Unit": 'ppb',
  "conv": 1000, # Conversão de ppm para ppb
  "tag":'CO',
  "Criteria_ave": 8,
  "criterias" : [9000],
  "criteria" : 9000,

  "Criteria_average": '8-h average',
}

O3 = {
  "Pollutant": "$O_{3}$",
  "Unit": 'ppm',
  "conv":1962 ,
  "tag":'O3',
  "Criteria_ave": 8,
  "criterias" : [140,130,120,100],
  "criteria" : 140,
  "Criteria_average": '8-h average',
}

SO2 = {
  "Pollutant": "$SO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 2620,
  "tag":'SO2',
  "Criteria_ave": 24,
  "criterias" : [125,50,40,30,20],
  "criteria" : 125,
  "Criteria_average": '24-h average',
  
}

PM10 = {
  "Pollutant": "$PM_{10}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM10',
  "Criteria_ave": 24,
  "criterias" : [120,100,75,50,45],
  "criteria" : 120,
  "Criteria_average": '24-h average',
}

PM25 = {
  "Pollutant": "$PM_{2.5}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM25',
  "Criteria_ave": 24,
  "criterias" : [60,50,37,25,15],
  "criteria" : 60,
  "Criteria_average": '24-h average',
}


pollutants = [NO2,O3,SO2,PM10,PM25]
#pollutants = [CO]

def cityTimeSeries(cityDataFrame,matData,cities,IBGE_CODE,cmap,legend,
               xlon,ylat,criteria,folder,pol,aveTime):
    import matplotlib.dates as mdates

    if len(matData.shape)==4:
        aveFigData= np.nanmean(matData,axis=0)[0,:,:]
    else:
        aveFigData= np.nanmean(matData,axis=0)

    if (np.nanmax(aveFigData)>0):

        cityArea=cities[cities['CD_MUN']==str(IBGE_CODE)]
        #cmap = plt.get_cmap(cmap,5)    
        fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [1, 3]})
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(14*cm, 7*cm)
        cmap.set_under('white')
        bounds = np.array([np.percentile(aveFigData[aveFigData>0],3),
                                 np.percentile(aveFigData[aveFigData>0],25),
                                 np.percentile(aveFigData[aveFigData>0],50),
                                 np.percentile(aveFigData[aveFigData>0],75),
                                 np.percentile(aveFigData[aveFigData>0],97),
                                 np.percentile(aveFigData[aveFigData>0],99.9)])
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        heatmap = ax[0].pcolor(xlon,ylat,aveFigData,cmap=cmap,norm=norm)
        cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                            ticks=bounds,
                            #extend='both',
                            spacing='uniform',
                            orientation='horizontal',
                            norm=norm,
                            ax=ax[0])

        cbar.ax.tick_params(rotation=30)
        #tick_locator = mpl.ticker.MaxNLocator(nbins=5)
        #cbar.locator = tick_locator
        #cbar.ax.set_xscale('log')
        #cbar.update_ticks()
        #cbar.ax.locator_params(axis='both',nbins=5)
        #cbar.ax.set_yscale('log')
        #cbar.update_ticks()
        #cbar.ax.set_xticklabels(['{:.1e}'.format(x) for x in bounds],rotation=30)
        cbar.ax.set_xlabel(cityArea['NM_MUN'].to_string(index=False)+'\nAverage', rotation=0,fontsize=6)
        cbar.ax.get_xaxis().labelpad = 5
        cbar.ax.tick_params(labelsize=6) 


        ax[0].set_xlim([cityArea.boundary.total_bounds[0],cityArea.boundary.total_bounds[2]])
        ax[0].set_ylim([cityArea.boundary.total_bounds[1],cityArea.boundary.total_bounds[3]])
        ax[0].set_frame_on(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        cityArea.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax[0])
        cities.boundary.plot(edgecolor='gray',linewidth=0.3,ax=ax[0])

        ax[1].fill_between(cityDataFrame.mean(axis=1).index,cityDataFrame.max(axis=1), cityDataFrame.min(axis=1),
                         color=cmap(0.8),       # The outline color
                         facecolor=cmap(0.8),
                         edgecolor=None,
                         alpha=0.2,label='Min-Max')          # Transparency of the fill
        ax[1].plot(cityDataFrame.mean(axis=1).index,cityDataFrame.mean(axis=1),
                   color=cmap(0.8),linewidth=1,label='Average')
        ax[1].xaxis.set_tick_params(labelsize=6)
        ax[1].yaxis.set_tick_params(labelsize=6)
        ax[1].set_ylim([np.nanmin(matData)*0.95,np.nanmax(matData)*1.05])
        ax[1].set_xlim([cityDataFrame.index.min(),cityDataFrame.index.max()])
        ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        # set formatter
        if criteria!=None:
            ax[1].axhline(y=criteria, color='yellow', linestyle='--',linewidth=1.5,
                          label='Air quality standard')
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        for label in ax[1].get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')
        ax[1].legend(prop={'size': 6})
        ax[1].set_ylabel(cityArea['NM_MUN'].to_string(index=False)+'\n'+legend,fontsize=6)
        ax[1].set_yscale('symlog')
 
        ax[1].yaxis.set_major_locator(plt.MaxNLocator(5))
        ax[1].grid(color='gray',linewidth = "0.2",axis='y')
        fig.tight_layout()
        fig.savefig(folder+'/cityTimeSeries_'+pol+'_'+aveTime+'.png', format="png",
                   bbox_inches='tight',dpi=300)
        plt.close(fig)
        return matData.shape



def cityTimeSeriesOnly(cityDataFrame,matData,cities,IBGE_CODE,cmap,legend,
               xlon,ylat,criteria,folder,pol,aveTime,criteriaVal):
    import matplotlib.dates as mdates

    if len(matData.shape)==4:
        aveFigData= np.nanmean(matData,axis=0)[0,:,:]
    else:
        aveFigData= np.nanmean(matData,axis=0)

    if (np.nanmax(aveFigData)>0):

        cityArea=cities[cities['CD_MUN']==str(IBGE_CODE)]
        #cmap = plt.get_cmap(cmap,5)    
        fig, ax = plt.subplots()
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(14*cm, 7*cm)
        # ax[1].fill_between(cityDataFrame.mean(axis=1).index,cityDataFrame.max(axis=1), cityDataFrame.min(axis=1),
        #                  color=cmap(0.8),       # The outline color
        #                  facecolor=cmap(0.8),
        #                  edgecolor=None,
        #                  alpha=0.2,label='Min-Max')          # Transparency of the fill
        ax.plot(cityDataFrame.mean(axis=1).index,cityDataFrame.mean(axis=1),
                   color=cmap(0.8),linewidth=1,label='Spatial average')
        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.set_ylim([np.nanmin(cityDataFrame.mean(axis=1))*0.95,
                     np.max(cityDataFrame.mean(axis=1).values,
                                )*1.05])
        ax.set_xlim([cityDataFrame.index.min(),cityDataFrame.index.max()])
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        # set formatter
        if (criteria!=None):
            if (criteriaVal<np.max(cityDataFrame.mean(axis=1).values)):
                ax.axhline(y=criteriaVal, color='yellow', linestyle='--',linewidth=1.5,
                              label='Air quality standard')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')
        ax.legend(prop={'size': 6},frameon=False)
        ax.set_ylabel(cityArea['NM_MUN'].to_string(index=False)+'\n'+legend,fontsize=6)
        #ax.set_yscale('log')
 
        #ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(color='gray',linewidth = "0.2",axis='y')
        fig.tight_layout()
        fig.savefig(folder+'/cityTimeSeriesONLY_'+pol+'_'+aveTime+'.png', format="png",
                   bbox_inches='tight',dpi=300)
        return matData.shape
#%%
def citiesBufferINdomain(xlon,ylat,cities,IBGE_CODE,atribute):
    s = gpd.GeoSeries(map(Point, zip(xlon.flatten(), ylat.flatten())))
    s = gpd.GeoDataFrame(geometry=s)
    s.crs = "EPSG:4326"
    s.to_crs("EPSG:4326")
    cities = cities.to_crs(epsg=4326)
    cityBuffer = cities[cities[atribute]==(IBGE_CODE)].buffer(0.5)
    cityBuffer.crs = "EPSG:4326"
    pointIn = cityBuffer.geometry.clip(s).explode()
    pointIn = gpd.GeoDataFrame({'geometry':pointIn}).reset_index()
    lia, loc = ismember.ismember(np.array((s.geometry.x,s.geometry.y)).transpose(),
                        np.array((pointIn.geometry.x,pointIn.geometry.y)).transpose(),'rows')
    s['city']=np.nan
    s.iloc[lia,1]=cities[atribute][pointIn['level_0'][loc]].values
    #s.iloc[lia,1]=1
    cityMat = np.reshape(np.array(s.city),(xlon.shape[0],xlon.shape[1])).astype(float)
    return s,cityMat,cityBuffer

def dataINcity(aveData,datesTime,cityMat,s,IBGE_CODE):
    #IBGE_CODE=4202404
    if np.size(aveData.shape)==4:
        cityData = aveData[:,:,cityMat==int(IBGE_CODE)]
        cityDataPoints = s[s.city.astype(float)==int(IBGE_CODE)]
        cityData = cityData[:,0,:]
        matData = aveData.copy()
        matData[:,:,cityMat!=int(IBGE_CODE)]=np.nan
        cityDataFrame=pd.DataFrame(cityData)
        cityDataFrame.columns = cityDataPoints.geometry.astype(str)
        cityDataFrame['Datetime']=datesTime.datetime
        cityDataFrame = cityDataFrame.set_index(['Datetime'])
    else:
        cityData = aveData[:,cityMat==int(1)]
        cityDataPoints = s[s.city.astype(float)==int(1)]
        cityData = cityData[:,:]
        matData = aveData.copy()
        matData[:,cityMat!=int(IBGE_CODE)]=np.nan
        cityDataFrame=pd.DataFrame(cityData)
        cityDataFrame.columns = cityDataPoints.geometry.astype(str)
        cityDataFrame['Datetime']=datesTime.datetime
        cityDataFrame = cityDataFrame.set_index(['Datetime'])
    return cityData,cityDataPoints,cityDataFrame,matData   

#pollutants=[CO]

#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
dataFolder = os.path.dirname(BASE)+'/data'
coarseDomainPath =  dataFolder+'/' + coarseDomain
refinedDomain =  dataFolder+'/' + refinedDomain
year = 2019

print('Looping for each variable')
for kk,pol in enumerate(pollutants):
    
    # ========BRAIN files============
    os.chdir(refinedDomain)
    print(pol)
    print('Openning netCDF files')
    # Opening netCDF files
    fileType='BRAIN_BASECONC_BR_'+pol['tag']+'_'+str(year)
    prefixed = sorted([filename for filename in os.listdir(refinedDomain) if filename.startswith(fileType)])
    ds = nc.MFDataset(prefixed)
    # Selecting variable
    dataBRAIN = ds[pol['tag']][:]*pol['conv']
    # Get datesTime and removing duplicates
    datesTimeBRAIN, dataBRAIN = BRAINutils.fixTimeBRAIN(ds,dataBRAIN)
    latBRAIN = ds['LAT'][:]
    lonBRAIN = ds['LON'][:]
    latBRAINflat = latBRAIN.flatten()
    lonBRAINflat = lonBRAIN.flatten()
    
    if pol['Criteria_ave']==1:
        aveData = dataBRAIN.copy()
    elif pol['Criteria_ave']==8:
        # Daily-maximum 8h-moving average
        aveData = tst.movingAverage(datesTimeBRAIN,dataBRAIN,8)
        datesTimeBRAIN = datesTimeBRAIN.groupby(by=['year', 'month', 'day']).size().reset_index()
        datesTimeBRAIN['datetime']=pd.to_datetime(datesTimeBRAIN[['year', 'month', 'day']])
    elif pol['Criteria_ave']==24:
        # Daily averages
        aveData, dailyData = tst.dailyAverage(datesTimeBRAIN,dataBRAIN)
        datesTimeBRAIN = datesTimeBRAIN.groupby(by=['year', 'month', 'day']).size().reset_index()
        datesTimeBRAIN['datetime']=pd.to_datetime(datesTimeBRAIN[['year', 'month', 'day']])   
    
    
    os.chdir(os.path.dirname(BASE))
    #capitals = pd.read_csv(os.path.dirname(BASE)+'/data/BR_capitais.csv')  
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Municipios_2020.shp'
    shape_path = os.path.dirname(os.path.dirname(BASE))+'/shapefiles/SC_Municipios_2022/SC_Municipios_2022.shp'

    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["azure","lightgray","pink","deeppink","purple"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["azure","lightgray","crimson","darkred"])
    legend = 'BRAIN ' +pol['Pollutant'] +' ('+ pol['Unit'] + ')'
    #legend ='BRAIN'
    cities = gpd.read_file(shape_path)
    cities.crs = "EPSG:4326"
    del ds, dataBRAIN
    os.makedirs(os.path.dirname(BASE)+'/figures/BRAINmunicipalities', exist_ok=True)
    os.makedirs(os.path.dirname(BASE)+'/tables'+'/'+pol['tag'], exist_ok=True)

    for IBGE_CODE in cities['CD_MUN']:
        IBGE_CODE=str(IBGE_CODE)
        s,cityMat,cityBuffer=citiesBufferINdomain(lonBRAIN,latBRAIN,cities,IBGE_CODE,'CD_MUN')
        #IBGE_CODE=1100205 #    
        cityData,cityDataPoints,cityDataFrame,matData= dataINcity(aveData,datesTimeBRAIN,cityMat,s,IBGE_CODE)
        
        cityTimeSeries(cityDataFrame,matData,cities,IBGE_CODE,cmap,legend,
                            lonBRAIN,latBRAIN,pol['criteria'],
                            os.path.dirname(BASE)+'/figures/BRAINmunicipalities/',pol['tag'],'BRAIN_'+str(IBGE_CODE))
        # cityTimeSeriesOnly(cityDataFrame,matData,cities,IBGE_CODE,cmap,legend,
        #                     lonBRAIN,latBRAIN,pol['Criteria_average'],
        #                     os.path.dirname(BASE)+'/figures/BRAINmunicipalities/',
        #                     pol['tag'],'BRAIN_'+str(IBGE_CODE),pol['Criteria'])
        cityDataFrame.to_csv(os.path.dirname(BASE)+'/tables'+'/'+pol['tag']+'/'+pol['tag']+'_'+str(IBGE_CODE)+'.csv')
        del s,cityMat,cityBuffer,cityData,cityDataPoints,cityDataFrame,matData
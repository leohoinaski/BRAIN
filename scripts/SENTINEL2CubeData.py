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
from scipy.interpolate import griddata
import matplotlib.pyplot as plt        
import geopandas as gpd
import matplotlib

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
#%%

def createNETCDFtemporalClipper(folderOut,name,data,ds,pollutant,xlon,ylat,datesTime):
    print('===================STARTING netCDFcreator_v1.py=======================')
    datesTime['TFLAG']=0
    for ii in range(0,data.shape[0]):
        datesTime['TFLAG'][ii] = np.int32(str(datesTime.year[ii])+\
            str(datesTime.month[ii]).zfill(2)+\
                str(datesTime.day[ii]).zfill(2)+\
                    str(datesTime.hour[ii]).zfill(2))
          
    f2 = nc.Dataset(folderOut+'/'+name,'w') #'w' stands for write 
    for gatr in ds.ncattrs() :
        print(gatr)
        try:
            setattr(f2, gatr, ds.__getattribute__(gatr))
        except:
            print('bad var')
    f2.NVARS= data.shape[1]
    f2.HISTORY =''
    setattr(f2, 'VAR-LIST', pollutant)
    f2.NVARS= 1
    f2.NCOLS = data.shape[2]
    f2.NROWS = data.shape[1]
    #f2.NVARS = data.shape[1]
    f2.SDATE = datesTime['TFLAG'][0]
    f2.FILEDESC = 'Concentration of ' +pollutant +' created by Leonardo Hoinaski - '
    f2.HISTORY = ''
    # # Specifying dimensions
    #tempgrp = f.createGroup('vehicularEmissions_data')
    f2.createDimension('TSTEP', None )
    f2.createDimension('DATE-TIME', 2)
    f2.createDimension('LAY', 1)
    f2.createDimension('VAR', 1)
    f2.createDimension('ROW', data.shape[1])
    f2.createDimension('COL', data.shape[2])
    # Building variables
    TFLAG = f2.createVariable('TFLAG', 'i4', ('TSTEP'))
    # Passing data into variables
    TFLAG[:] = datesTime['TFLAG']
    LON = f2.createVariable('LON', 'f4', ( 'ROW','COL'))
    LAT = f2.createVariable('LAT', 'f4', ( 'ROW','COL'))
    LAT[:,:] =  ylat
    LON[:,:] = xlon
    LON.units = 'degrees '
    LAT.units = 'degrees '
    globals()[pollutant] = f2.createVariable(pollutant, np.float32, ('TSTEP', 'LAY', 'ROW','COL'))
    globals()[pollutant][:,0,:,:] = data[:,:,:]
    globals()[pollutant].units = ds[pollutant].units
    f2.close()
    return f2

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

# #%%
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
    prefixed = sorted([filename for filename in os.listdir(
             coarseDomainPath) if filename.startswith('SENTINEL_'+pol['tag'])])

    ds2 = nc.Dataset(coarseDomainPath+'/'+prefixed[0])
    matAve = ds2[pol['tag']][:]
#     os.chdir(coarseDomainPath+'/'+pol['tag'])
#     print('Openning netCDF files')
#     # Opening netCDF files
#     fileType = 'S5P_OFFL_L2__'+pol['tag']
#     prefixed = sorted([filename for filename in os.listdir(
#         coarseDomainPath+'/'+pol['tag']) if filename.startswith(fileType)])

#     matAve=np.empty((dailyData.shape[0],latBRAIN.shape[0],latBRAIN.shape[1]))
#     matAve[:,:,:] = np.nan
#     times=[]
    
#     for pr in prefixed:
#         ds2 = nc.Dataset(pr)
#         dataInBRAIN = np.empty(lonBRAINflat.shape[0])
#         dataInBRAIN[:] = np.nan
        
#         try:
#             time = ds2.groups['PRODUCT'].variables['time'][:]
#             time = datetime.datetime.fromtimestamp(
#                 tinit.timestamp()+time[0]).strftime('%Y-%m-%d 00:00:00')
#             print(pr)
#             print(time)
#             times.append(time)
        
#             if (daily.datetime ==time).sum()==1:
                
 
#                 # print (ds2.groups['PRODUCT'].variables.keys())
#                 # print (ds2.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'])
#                 lonsOriginal = ds2.groups['PRODUCT'].variables['longitude'][:][0,:, :].data.flatten()
#                 lonsOriginal[(lonsOriginal>180) | (lonsOriginal<-180)]=np.nan
                
#                 latsOriginal = ds2.groups['PRODUCT'].variables['latitude'][:][0,:, :].data.flatten()
#                 latsOriginal[(latsOriginal>90) | (latsOriginal<-90)]=np.nan
                
#                 lons= lonsOriginal.copy()
#                 lons[(latsOriginal>np.nanmax(latBRAINflat)) |
#                      (latsOriginal<np.nanmin(latBRAINflat)) |
#                      (lonsOriginal>np.nanmax(lonBRAINflat)) |
#                      (lonsOriginal<np.nanmin(lonBRAINflat)) ]=np.nan
                
#                 lats= latsOriginal.copy()
#                 lats[(latsOriginal>np.nanmax(latBRAINflat)) |
#                      (latsOriginal<np.nanmin(latBRAINflat)) |
#                      (lonsOriginal>np.nanmax(lonBRAINflat)) |
#                      (lonsOriginal<np.nanmin(lonBRAINflat)) ]=np.nan
#                 if pol['tag'] =='NO2':
#                     dataSentinelOriginal = ds2.groups['PRODUCT'].variables['nitrogendioxide_tropospheric_column'][0, :,:].data.flatten()
#                 elif pol['tag'] =='CO':
#                     dataSentinelOriginal = ds2.groups['PRODUCT'].variables['carbonmonoxide_total_column'][0, :,:].data.flatten()

                    
#                 dataSentinel = dataSentinelOriginal[~np.isnan(lats)]
#                 lats = lats[~np.isnan(lats)]
#                 lons = lons[~np.isnan(lons)]
#                 lats = lats[dataSentinel!=9.96921e+36]
#                 lons = lons[dataSentinel!=9.96921e+36]
#                 dataSentinel = dataSentinel[dataSentinel!=9.96921e+36]
       
#                 print('-----FILE OK!!-----')
#                 #print(time)
     
             
#                 grid_z0 = griddata(np.array([lons,lats]).transpose(), 
#                                    dataSentinel, (lonBRAIN, latBRAIN), 
#                                    method='linear',fill_value=np.nan, rescale=True)
#                 #grid_z0 = griddata(np.array([lonsOriginal,latsOriginal]).transpose(), 
#                 #                   dataSentinelOriginal, (lonBRAIN, latBRAIN), method='nearest')
             
#                 matAve[daily.datetime ==time,:,:] = \
#                     np.nanmean([matAve[daily.datetime ==time,:,:], 
#                                grid_z0.reshape(1,matAve.shape[1],matAve.shape[2])],axis=0)
            
            
#         except:
#             print(pr)
#             print('file without data')
    
#     name = 'SENTINEL_'+pol['tag']+'_'+str(daily.datetime[0])+'_'+str(daily.iloc[-1].datetime)
#     createNETCDFtemporalClipper(coarseDomainPath,name,matAve,ds,pol['tag'],lonBRAIN,latBRAIN,daily)




#%%
def brainPcolor(BASE,pol,lonBRAIN,latBRAIN,dataBRAIN,
                lonMERRA,latMERRA,dataMERRA,borda):

    fig,ax = plt.subplots(1,2)
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(10*cm, 7*cm)

    heatmap = ax[0].pcolor(lonBRAIN,latBRAIN,dataBRAIN*pol['conv'],
                            #vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                            #vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]),
                            #norm=matplotlib.colors.LogNorm(),
                            #norm=norm,
                            #    vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])])*1.2,
                            #    vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])])*0.8),
                            cmap='Spectral_r')
        

    ax[0].set_xlim([lonBRAIN.min()+1, lonBRAIN.max()])
    ax[0].set_ylim([latBRAIN.min(), latBRAIN.max()-1])
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',
                        #norm=matplotlib.colors.LogNorm(vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                        #                               vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]))
                        )
    #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
    cbar.ax.set_xlabel(pol['tag']+' ('+pol['Unit']+')\n a) BRAIN original', rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 2
    cbar.ax.tick_params(labelsize=6)
    
  

    heatmap = ax[1].pcolor(lonMERRA,latMERRA,dataMERRA,
                            #norm=matplotlib.colors.LogNorm(),
                                #vmin=np.nanpercentile(dataMERRA,1),
                                #vmax=np.nanpercentile(dataMERRA,90)),
                            cmap='Spectral_r')
    ax[1].set_xlim([lonBRAIN.min()+1, lonBRAIN.max()])
    ax[1].set_ylim([latBRAIN.min(), latBRAIN.max()-1])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',
                        #norm=matplotlib.colors.LogNorm(vmin=np.nanpercentile(dataMERRA,1),
                        #                                vmax=np.nanpercentile(dataMERRA,95)),
                        )
    #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
    cbar.ax.set_xlabel(pol['tag']+' tropospheric column ($10^{-4} mol.m^{-2}$) \n b) Regrided SENTINEL/TROPOMI', rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 2
    cbar.ax.tick_params(labelsize=6)
 
    
    borda.boundary.plot(ax=ax[0],edgecolor='black',linewidth=0.3)
    borda.boundary.plot(ax=ax[1],edgecolor='black',linewidth=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsSENTINEL_average_'+pol['tag']+'.png', 
                format="png",bbox_inches='tight',dpi=300)


matAve2 = matAve[:,0,:,:].copy()
matAve2[matAve2<0]=np.nan
matAve2[matAve2>1]=np.nan
matAve3 = matAve[:,0,:,:].copy()
matAve3[matAve3<0]=np.nan
matAve3[matAve3>10]=np.nan

shapeBorder = '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
borda = gpd.read_file(shapeBorder)
brainPcolor(BASE,pol,lonBRAIN,latBRAIN,np.nanmean(dailyData[:,0,:,:],axis=0),
                lonBRAIN,latBRAIN,(10**4)*np.nanmean(matAve3[:,:,:],axis=0),borda)

np.nansum(matAve)
#%%

from shapely.geometry import Point
import ismember  
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
    
import geopandas as gpd
import scipy
def BRAINscattersRegions(shape_path,BASE,pol,xvMERRA,yvMERRA,dataMERRAfiltered,
                         dataBRAINinMERRA,bordAtrib):
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_regions.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
    dataShp = gpd.read_file(shape_path)
        
    for sigla in dataShp[bordAtrib].values:
        uf = dataShp[dataShp[bordAtrib]==sigla]
        xlon,ylat = xvMERRA,yvMERRA
        
        if sigla == 'Sul':
            sigla='South'
            c='Blue'
        elif sigla=='Norte':
            sigla='North'
            c='Red'
        elif sigla=='Sudeste':
            sigla='Southeast'
            c='Gray'
        elif sigla=='Nordeste':
            sigla='Northeast'
            c='Orange'
        elif sigla=='Centro-Oeste':
            sigla='Midwest'
            c='Brown'
        if sigla == 'Brasil':
            sigla='Brazil'
            c='gold'

   
        s,cityMat=dataINshape(xlon,ylat,uf)
        #plt.pcolor(cityMat)

        fig,ax = plt.subplots()
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(7*cm, 7*cm)
        xy = np.vstack([dataMERRAfiltered[:,cityMat==1].flatten(),dataBRAINinMERRA[:,cityMat==1].flatten()])
        xy = xy[:,~np.any(np.isnan(xy), axis=0)]
        #z = gaussian_kde(xy)(xy)
        ax.scatter(xy[0,:],xy[1,:],
                    s=1,alpha=.2,c=c)
    
        ###calculate Spearman correlation using new_df
        corr, p_value = scipy.stats.spearmanr(xy[0,:],xy[1,:])
       
        ###insert text with Spearman correlation
        # ax.annotate('ρ = {:.2f}'.format(corr), 
        #         xy=(0.70, 0.9), xycoords='axes fraction', 
        #         fontsize=8, ha='left', va='center')
 
        ax.annotate(sigla+'\nρ = {:.2f}'.format(corr),
                xy=(0.57, 0.1), xycoords='axes fraction', 
                fontsize=8, ha='left', va='center')
        
        y,preds = xy[0,:],xy[1,:]
        
        ax.set_xlim([np.nanmin([y]),np.nanmax([y])])
        ax.set_ylim([np.nanmin([preds]),np.nanmax([preds])])
        ax.set_xlabel('SENTINEL/TROPOMI\n'+pol['tag']+' tropospheric column \n ($10^{-4}  mol.m^{-2}$)',fontsize=6)
        ax.set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
        ax.xaxis.set_tick_params(labelsize=7)
        ax.yaxis.set_tick_params(labelsize=8)
            
        ax.set_yscale('log')
        ax.set_xscale('log')
        fig.tight_layout()
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsSENTINEL_scatter_'+sigla+'_'+pol['tag']+'.png', 
                    format="png",bbox_inches='tight',dpi=300)
        
        
    
    # BRASIL TODO    
    fig,ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(7*cm, 7*cm)
    xy = np.vstack([dataMERRAfiltered.flatten(),dataBRAINinMERRA.flatten()])
    xy = xy[:,~np.any(np.isnan(xy), axis=0)]
    #z = gaussian_kde(xy)(xy)
    ax.scatter(xy[0,:],xy[1,:],
                s=1,alpha=.2,c='Cyan')
    sigla='Domain'
    ###calculate Spearman correlation using new_df
    corr, p_value = scipy.stats.spearmanr(xy[0,:],xy[1,:])
   
    ###insert text with Spearman correlation
    # ax.annotate('ρ = {:.2f}'.format(corr), 
    #         xy=(0.70, 0.9), xycoords='axes fraction', 
    #         fontsize=8, ha='left', va='center')

    ax.annotate(sigla+'\nρ = {:.2f}'.format(corr),
            xy=(0.57, 0.1), xycoords='axes fraction', 
            fontsize=8, ha='left', va='center')
    
    y,preds = xy[0,:],xy[1,:]



    ax.set_xlim([np.nanmin([y]),np.nanmax([y])])
    ax.set_ylim([np.nanmin([preds]),np.nanmax([preds])])
    ax.set_xlabel('SENTINEL/TROPOMI\n'+pol['tag']+' tropospheric column \n ($10^{-4}  mol.m^{-2}$)',fontsize=6)
    ax.set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsSENTINEL_scatter_'+sigla+'_'+pol['tag']+'.png', 
                format="png",bbox_inches='tight',dpi=300)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    
#shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp' 
shape_path= '/media/leohoinaski/HDD/shapefiles/BR_regions.shp'   

#bordAtrib='NM_PAIS'
bordAtrib='NM_MUN'
BRAINscattersRegions(shape_path,BASE,pol,lonBRAIN,latBRAIN,(10**4)*matAve2,dailyData[:,0,:,:],bordAtrib)

shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp' 
bordAtrib='NM_PAIS'
BRAINscattersRegions(shape_path,BASE,pol,lonBRAIN,latBRAIN,(10**4)*matAve2,dailyData[:,0,:,:],bordAtrib)

#%%
def dataINcity(aveData,datesTime,cityMat,s,IBGE_CODE):
    #IBGE_CODE=4202404
    if np.size(aveData.shape)==4:
        cityData = aveData[:,:,cityMat==IBGE_CODE]
        cityDataPoints = s[s.city.astype(float)==IBGE_CODE]
        cityData = cityData[:,0,:]
        matData = aveData.copy()
        matData[:,:,cityMat!=IBGE_CODE]=np.nan
        cityDataFrame=pd.DataFrame(cityData)
        cityDataFrame.columns = cityDataPoints.geometry.astype(str)
        cityDataFrame['Datetime']=datesTime.datetime
        cityDataFrame = cityDataFrame.set_index(['Datetime'])
    else:
        cityData = aveData[:,cityMat==int(IBGE_CODE)]
        cityDataPoints = s[s.city.astype(float)==int(IBGE_CODE)]
        cityData = cityData[:,:]
        matData = aveData.copy()
        matData[:,cityMat!=int(IBGE_CODE)]=np.nan
        cityDataFrame=pd.DataFrame(cityData)
        cityDataFrame.columns = cityDataPoints.geometry.astype(str)
        cityDataFrame['Datetime']=datesTime.datetime
        cityDataFrame = cityDataFrame.set_index(['Datetime'])
    return cityData,cityDataPoints,cityDataFrame,matData   

def citiesBufferINdomain(xlon,ylat,cities,IBGE_CODE):
    s = gpd.GeoSeries(map(Point, zip(xlon.flatten(), ylat.flatten())))
    s = gpd.GeoDataFrame(geometry=s)
    s.crs = "EPSG:4326"
    s.to_crs("EPSG:4326")
    cities = cities.to_crs(epsg=4326)
    cityBuffer = cities[cities['CD_MUN']==(IBGE_CODE)].buffer(0.5)
    cityBuffer.crs = "EPSG:4326"
    pointIn = cityBuffer.geometry.clip(s).explode()
    pointIn = gpd.GeoDataFrame({'geometry':pointIn}).reset_index()
    lia, loc = ismember.ismember(np.array((s.geometry.x,s.geometry.y)).transpose(),
                        np.array((pointIn.geometry.x,pointIn.geometry.y)).transpose(),'rows')
    s['city']=np.nan
    s.iloc[lia,1]=cities['CD_MUN'][pointIn['level_0'][loc]].values
    cityMat = np.reshape(np.array(s.city),(xlon.shape[0],xlon.shape[1])).astype(float)
    return s,cityMat,cityBuffer

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
        xdata = [datetime.datetime.strptime(x, '%Y-%m-%d 00:00:00') for x in cityDataFrame.mean(axis=1).dropna().index]
        ax[1].fill_between(xdata,cityDataFrame.max(axis=1).dropna(), cityDataFrame.min(axis=1).dropna(),
                         color=cmap(0.8),       # The outline color
                         facecolor=cmap(0.8),
                         edgecolor=None,
                         alpha=0.2,label='Min-Max')          # Transparency of the fill
        ax[1].plot(xdata,cityDataFrame.mean(axis=1).dropna(),
                   color=cmap(0.8),linewidth=1,label='Average')
        ax[1].xaxis.set_tick_params(labelsize=6)
        ax[1].yaxis.set_tick_params(labelsize=6)
        ax[1].set_ylim([np.nanmin(matData)*0.95,np.nanmax(matData)*1.05])
        ax[1].set_xlim([np.min(xdata),np.max(xdata)])
        ax[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        # set formatter
        if criteria!=None:
            ax[1].axhline(y=criteria, color='gray', linestyle='--',linewidth=0.5,
                          label='Air quality standard')
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        for label in ax[1].get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')
        ax[1].legend(prop={'size': 6})
        ax[1].set_ylabel(cityArea['NM_MUN'].to_string(index=False)+'\n'+legend,fontsize=6)
        fig.tight_layout()
        fig.savefig(folder+'/cityTimeSeries_'+pol+'_'+aveTime+'.png', format="png",
                   bbox_inches='tight',dpi=300)
        return matData.shape


capitals = pd.read_csv(os.path.dirname(BASE)+'/data/BR_capitais.csv')  

shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Municipios_2020.shp'

cities = gpd.read_file(shape_path)
cities.crs = "EPSG:4326"
aveData = dailyData[:,0,:,:]
aveData2 = (10**4)*matAve2
datesTime = daily
xlon,ylat =lonBRAIN,latBRAIN 

#cmap = 'YlOrRd'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["azure","cornsilk","yellow","darkred"])

legend = 'SENTINEL/TROPOMI \n' +pol['Pollutant'] +' tropospheric column \n ($10^{-4}  mol.m^{-2}$)'
#legend ='BRAIN'
for IBGE_CODE in capitals.IBGE_CODE:
    IBGE_CODE=str(IBGE_CODE)
    s,cityMat,cityBuffer=citiesBufferINdomain(xlon,ylat,cities,IBGE_CODE)
    #IBGE_CODE=1100205 #    
    cityData,cityDataPoints,cityDataFrame,matData= dataINcity(aveData2,datesTime,cityMat,s,IBGE_CODE)
    cityTimeSeries(cityDataFrame,matData,cities,IBGE_CODE,cmap,legend,
                        xlon,ylat,None,
                        os.path.dirname(BASE)+'/figures/',pol['tag'],'SENTINEL_'+str(IBGE_CODE))
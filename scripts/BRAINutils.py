#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:48:15 2024

@author: leohoinaski
"""
#import os
import numpy as np
from datetime import datetime
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Point
import ismember  
#from numpy.lib.stride_tricks import sliding_window_view
#import pyproj
#from shapely.geometry import Point
#import geopandas as gpd
#from ismember import ismember
#import wrf
import os
import matplotlib.pyplot as plt
import scipy
import matplotlib

def datePrepBRAIN(ds):
    tf = np.array(ds['TFLAG'][:])
    date=[]
    for ii in range(0,tf.shape[0]):
        date.append(datetime.strptime(tf[:].astype(str)[ii], '%Y%m%d%H').strftime('%Y-%m-%d %H:00:00'))
    
    date = np.array(date,dtype='datetime64[s]')
    dates = pd.DatetimeIndex(date)
    datesTime=pd.DataFrame()
    datesTime['year'] = dates.year
    datesTime['month'] = dates.month
    datesTime['day'] = dates.day
    datesTime['hour'] = dates.hour
    datesTime['datetime']=dates
    return datesTime

def fixTimeBRAIN(ds,data):
    dd = datePrepBRAIN(ds)
    idx2Remove = np.array(dd.drop_duplicates().index)
    data = data[idx2Remove[0:(data.shape[0])]]
    datesTime = dd.drop_duplicates().reset_index(drop=True)
    datesTime=datesTime[0:(data.shape[0])]
    return datesTime,data

def fixTimeBRAINemis(ds,data):
    dd = datePrepBRAINemis(ds)
    idx2Remove = np.array(dd.drop_duplicates().index)
    data = data[idx2Remove[0:(data.shape[0])]]
    datesTime = dd.drop_duplicates().reset_index(drop=True)
    datesTime=datesTime[0:(data.shape[0])]
    return datesTime,data

def datePrepBRAINemis(ds):
    tf = np.array(ds['TFLAG'][0:8759])
    date=[]
    for ii in range(0,tf.shape[0]):
        date.append(datetime.strptime(tf[:].astype(str)[ii], '%Y%m%d%H').strftime('%Y-%m-%d %H:00:00'))
    
    date = np.array(date,dtype='datetime64[s]')
    dates = pd.DatetimeIndex(date)
    datesTime=pd.DataFrame()
    datesTime['year'] = dates.year
    datesTime['month'] = dates.month
    datesTime['day'] = dates.day
    datesTime['hour'] = dates.hour
    datesTime['datetime']=dates
    return datesTime

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
  

def dataINcity(aveData,datesTime,cityMat,s,IBGE_CODE):
    #IBGE_CODE=4202404
    if np.size(aveData.shape)==4:
        cityData = aveData[:,:,cityMat==1]
        cityDataPoints = s[s.city.astype(float)==1]
        cityData = cityData[:,0,:]
        matData = aveData.copy()
        matData[:,:,cityMat!=IBGE_CODE]=np.nan
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

def citiesBufferINdomain(xlon,ylat,cities,IBGE_CODE,atribute):
    s = gpd.GeoSeries(map(Point, zip(xlon.flatten(), ylat.flatten())))
    s = gpd.GeoDataFrame(geometry=s)
    s.crs = "EPSG:4326"
    s.to_crs("EPSG:4326")
    cities = cities.to_crs(epsg=4326)
    cityBuffer = cities[cities[atribute]==(IBGE_CODE)].buffer(0.2)
    cityBuffer.crs = "EPSG:4326"
    pointIn = cityBuffer.geometry.clip(s).explode()
    pointIn = gpd.GeoDataFrame({'geometry':pointIn}).reset_index()
    lia, loc = ismember.ismember(np.array((s.geometry.x,s.geometry.y)).transpose(),
                        np.array((pointIn.geometry.x,pointIn.geometry.y)).transpose(),'rows')
    s['city']=np.nan
    #s.iloc[lia,1]=cities[atribute][pointIn['level_0'][loc]].values
    s.iloc[lia,1]=1
    cityMat = np.reshape(np.array(s.city),(xlon.shape[0],xlon.shape[1])).astype(float)
    return s,cityMat,cityBuffer

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

def dailyAverageSentinel (datesTime,data):
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

def brainPcolor(BASE,pol,polSentinel,lonBRAIN,latBRAIN,dataBRAIN,
                lonMERRA,latMERRA,dataMERRA,borda):

    fig,ax = plt.subplots(1,2)
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(18*cm, 12*cm)

    heatmap = ax[0].pcolor(lonBRAIN,latBRAIN,dataBRAIN,
                            #vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                            #vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]),
                            norm=matplotlib.colors.LogNorm(),
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
                        norm=matplotlib.colors.LogNorm(),
                        #norm=matplotlib.colors.LogNorm(vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                        #                               vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]))
                        )

    #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
    cbar.ax.set_xlabel(pol['tag']+' ('+pol['Unit']+')\n a) BRAIN original', rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 2
    cbar.ax.tick_params(labelsize=6)
    #cbar.ax.set_major_locator(plt.MaxNLocator(5))

    heatmap = ax[1].pcolor(lonMERRA,latMERRA,dataMERRA,
                            norm=matplotlib.colors.LogNorm(),
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
                        norm=matplotlib.colors.LogNorm(),
                        #norm=matplotlib.colors.LogNorm(vmin=np.nanpercentile(dataMERRA,1),
                        #                                vmax=np.nanpercentile(dataMERRA,95)),
                        )
    #cbar.ax.set_major_locator(plt.MaxNLocator(5))
    
    
    #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
    cbar.ax.set_xlabel(polSentinel['tag']+' tropospheric column ($10^{-4} mol.m^{-2}$) \n b) Regrided SENTINEL/TROPOMI', rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 2
    cbar.ax.tick_params(labelsize=6)
    cbar.update_ticks()
    
    borda.boundary.plot(ax=ax[0],edgecolor='black',linewidth=0.3)
    borda.boundary.plot(ax=ax[1],edgecolor='black',linewidth=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsSENTINEL_average_'+pol['tag']+'.png', 
                format="png",bbox_inches='tight',dpi=300)

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
        #print([np.nanmin([y]),np.nanmax([y])])
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
        
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:01:28 2024

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
coarseDomain = 'MERRA'
refinedDomain = 'BRAIN' 

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

pollutants=[CO]

#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
dataFolder = os.path.dirname(BASE)+'/data'
coarseDomainPath =  dataFolder+'/' + coarseDomain
refinedDomain =  dataFolder+'/' + refinedDomain


print('Looping for each variable')
for kk,pol in enumerate(pollutants):
    
    # ========BRAIN files============
    os.chdir(refinedDomain)
    print(pol)
    print('Openning netCDF files')
    # Opening netCDF files
    fileType='BRAIN_BASECONC_BR_'+pol['tag']
    prefixed = sorted([filename for filename in os.listdir(refinedDomain) if filename.startswith(fileType)])
    ds = nc.MFDataset(prefixed)
    # Selecting variable
    dataBRAIN = ds[pol['tag']][:]
    # Get datesTime and removing duplicates
    datesTimeBRAIN, dataBRAIN = BRAINutils.fixTimeBRAIN(ds,dataBRAIN)
    latBRAIN = ds['LAT'][:]
    lonBRAIN = ds['LON'][:]
    latBRAINflat = latBRAIN.flatten()
    lonBRAINflat = lonBRAIN.flatten()
    os.chdir(os.path.dirname(BASE))
    # ========coarse domain files============
    os.chdir(coarseDomainPath)
    print('Openning netCDF files')
    # Opening netCDF files
    fileType='MERRA2_400'
    prefixed = sorted([filename for filename in os.listdir(coarseDomainPath) if filename.startswith(fileType)])
    ds = xr.open_mfdataset(prefixed)
    os.chdir(os.path.dirname(BASE))
    # Selecting variable
    if pol['tag']=='CO':
        polMERRA = 'COSC'
        dataMERRA = ds[polMERRA][:]
    elif pol['tag']=='O3':
        polMERRA = 'TO3'
        dataMERRA = ds[polMERRA][:]
    elif pol['tag']=='SO2':
        polMERRA = 'SO2SMASS'
        dataMERRA = ds[polMERRA][:]*(10**9)
    elif pol['tag'] == 'PM25':
        # dust25 = ds['DUSMASS25'][:]
        # ss25 = ds['SSSMASS25'][:]
        # bc = ds['BCSMASS'][:]
        # oc = ds['OCSMASS'][:]
        # so4 = ds['SO4SMASS'][:]
        dataMERRA = (ds['DUSMASS25'][:]+ ds['SSSMASS25'][:]+ ds['BCSMASS'][:] +\
                     1.8*ds['OCSMASS'][:]+ 1.375*ds['SO4SMASS'][:])*(10**9)

    
    #dataMERRA = ds[polMERRA][:]
    timeMerra = ds['time'].time.data
    timeMerra = pd.DatetimeIndex(timeMerra)
    datesTimeMERRA=pd.DataFrame()
    datesTimeMERRA['year'] = timeMerra.year
    datesTimeMERRA['month'] = timeMerra.month
    datesTimeMERRA['day'] = timeMerra.day
    datesTimeMERRA['hour'] = timeMerra.hour
    datesTimeMERRA['datetime']=timeMerra
    datesTimeMERRA['datetime']=  datesTimeMERRA['datetime'].dt.strftime('%Y-%m-%d %H:00:00')
    latMERRA = ds['lat'].data[:]
    lonMERRA = ds['lon'].data[:]
    xvMERRA,yvMERRA =np.meshgrid(lonMERRA,latMERRA)
    latMERRAflat= yvMERRA.flatten()
    lonMERRAflat = xvMERRA.flatten()
    
    pixBRAINinMERRA=[]
    for ii,lats in enumerate(latBRAINflat):
        print(ii)
        if (lats>np.max(latMERRAflat)) and (lats<np.max(latMERRAflat)) and \
            (lonBRAINflat[ii]>np.max(lonMERRAflat)) and (lonBRAINflat[ii]<np.max(lonMERRAflat)):
            pixBRAINinMERRA.append(np.nan)  
        else:
            dist = np.sqrt((lats-latMERRAflat)**2+(lonBRAINflat[ii]-lonMERRAflat)**2)
            pixBRAINinMERRA.append(np.argmin(abs(dist)))
    
    pixMERRAunique = np.unique(pixBRAINinMERRA)
    latsIn = np.unique(latMERRAflat[pixMERRAunique])
    lonsIn = np.unique(lonMERRAflat[pixMERRAunique])
    
    matAveMERRA=np.empty((dataBRAIN.shape[0],latMERRAflat.shape[0]))
    matAveMERRA[:,:] = numpy.nan
    matAveLat=[]
    matAveLon=[]
    for ii,pixMERRA in enumerate(pixMERRAunique):
        latAve =latBRAINflat[pixBRAINinMERRA==pixMERRA]
        lonAve =lonBRAINflat[pixBRAINinMERRA==pixMERRA]
        
        matAve = np.zeros([dataBRAIN.shape[0],latAve.shape[0]])

        for jj,latIN in enumerate(latAve):
            matAve[:,jj]=dataBRAIN[:,:,latBRAIN[:,0]==latIN,lonBRAIN[0,:]==lonAve[jj]][:,0,0]
            print(str(ii)+'  '+str(jj))
            
        matAveMERRA[:,pixMERRA] = np.nanmedian(matAve,axis=1) 
        matAveLat.append(latMERRAflat[pixMERRA])
        matAveLon.append(lonMERRAflat[pixMERRA])
        
        
    dataBRAINinMERRA = matAveMERRA.reshape(matAveMERRA.shape[0],latMERRA.shape[0],lonMERRA.shape[0])
    
    I, idx = ismember.ismember(datesTimeMERRA.datetime.astype(str),
                               datesTimeBRAIN.datetime.astype(str))
    
    dataBRAINinMERRA = dataBRAINinMERRA[idx,:,:]*pol['conv']
    dataMERRAfiltered = np.array(dataMERRA[I,:,:])
    
    dataMERRAfiltered[np.isnan(dataBRAINinMERRA)] = np.nan
    dataBRAIN=dataBRAIN[idx,:,:,:]
    
    
    
    shapeBorder = '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
    borda = gpd.read_file(shapeBorder)


    #%%
    if pol['tag']=='O3':
        fig,ax = plt.subplots(1,3)
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(18*cm, 7*cm)
    else:
        fig,ax = plt.subplots(1,4)
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(18*cm, 7*cm)
    
    xb = [np.nanmin(xvMERRA[~np.isnan(dataBRAINinMERRA)[0,:,:]]),
          np.nanmax(xvMERRA[~np.isnan(dataBRAINinMERRA)[0,:,:]])]
    yb = [np.nanmin(yvMERRA[~np.isnan(dataBRAINinMERRA)[0,:,:]]),
          np.nanmax(yvMERRA[~np.isnan(dataBRAINinMERRA)[0,:,:]])]
    
    if pol['tag']=='O3':
        heatmap = ax[0].pcolor(lonBRAIN,latBRAIN,dataBRAIN[500,0,:,:]*pol['conv'],
                                #vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                #vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]),
                                #norm=matplotlib.colors.LogNorm(
                                #    vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])])*1.2,
                                #    vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])])*0.8),
                                cmap='Spectral_r')
        
    else:
        heatmap = ax[0].pcolor(lonBRAIN,latBRAIN,dataBRAIN[500,0,:,:]*pol['conv'],
                                vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]),
                                cmap='Spectral_r')
        
    ax[0].set_xlim([xb[0], xb[1]])
    ax[0].set_ylim([yb[0], yb[1]])
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
    
    if pol['tag']=='O3':
        heatmap1 = ax[1].pcolor(lonMERRA,latMERRA,dataBRAINinMERRA[500,:,:],
                                #vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                #vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]),
                                #norm=matplotlib.colors.LogNorm(
                                #   vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])])*1.2,
                                #   vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])])*0.8),
                                cmap='Spectral_r')
    else:
        heatmap1 = ax[1].pcolor(lonMERRA,latMERRA,dataBRAINinMERRA[500,:,:],
                                vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]),
                                cmap='Spectral_r')    
        
    ax[1].set_xlim([xb[0], xb[1]])
    ax[1].set_ylim([yb[0], yb[1]])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    cbar = fig.colorbar(heatmap1,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',
                        #norm=matplotlib.colors.LogNorm(vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                        #                               vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]))
                        cmap='Spectral_r')
    #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
    cbar.ax.set_xlabel(pol['tag']+' ('+pol['Unit']+')\n b) BRAIN regrid', rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 2
    cbar.ax.tick_params(labelsize=6)
    
    if pol['tag']=='O3':
        heatmap = ax[2].pcolor(lonMERRA,latMERRA,dataMERRAfiltered[500,:,:],
                                #norm=matplotlib.colors.LogNorm(
                                #    vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                #    vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])])),
                                cmap='Spectral_r')
        ax[2].set_xlim([xb[0], xb[1]])
        ax[2].set_ylim([yb[0], yb[1]])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                            #extend='both', 
                            #ticks=bounds,
                            #spacing='uniform',
                            orientation='horizontal',
                            norm=matplotlib.colors.LogNorm(vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                                            vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])])),
                            )
        #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
        cbar.ax.set_xlabel(pol['tag']+' (Dobsons)\n c) MERRA2', rotation=0,fontsize=6)
        cbar.ax.get_xaxis().labelpad = 2
        cbar.ax.tick_params(labelsize=6)
    else:
        heatmap = ax[2].pcolor(lonMERRA,latMERRA,dataMERRAfiltered[500,:,:],                               
                                cmap='Spectral_r')
        ax[2].set_xlim([xb[0], xb[1]])
        ax[2].set_ylim([yb[0], yb[1]])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                            #extend='both', 
                            #ticks=bounds,
                            #spacing='uniform',
                            orientation='horizontal',
                            norm=matplotlib.colors.LogNorm(vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                                            vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])])),
                            )
        #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
        cbar.ax.set_xlabel(pol['tag']+' ('+pol['Unit']+')\n c) MERRA2', rotation=0,fontsize=6)
        cbar.ax.get_xaxis().labelpad = 2
        cbar.ax.tick_params(labelsize=6)
    
    if pol['tag']!='O3':
        heatmap = ax[3].pcolor(lonMERRA,latMERRA,
                      np.nanmean(dataMERRAfiltered-dataBRAINinMERRA,axis=0),
                      vmin=-np.nanmax(np.nanmean(dataMERRAfiltered-dataBRAINinMERRA,axis=0))*0.9,
                      vmax=np.nanmax(np.nanmean(dataMERRAfiltered-dataBRAINinMERRA,axis=0))*0.9,
                      #vmin=np.nanmin(np.nanmean(dataMERRAfiltered-dataBRAINinMERRA)),
                      #vmax=abs(np.nanmin(np.nanmean(dataMERRAfiltered-dataBRAINinMERRA))),
                      cmap='RdBu_r')
        ax[3].set_xlim([xb[0], xb[1]])
        ax[3].set_ylim([yb[0], yb[1]])
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        borda.boundary.plot(ax=ax[3],edgecolor='black',linewidth=0.3)
        cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                            #extend='both', 
                            #ticks=bounds,
                            #spacing='uniform',
                            orientation='horizontal',
                            #norm=matplotlib.colors.LogNorm()
                            )
        #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
        cbar.ax.set_xlabel(pol['tag']+' ('+pol['Unit']+')\n d) MERRA2 - BRAIN regrid', rotation=0,fontsize=6)
        cbar.ax.get_xaxis().labelpad = 2
        cbar.ax.tick_params(labelsize=6)
    
    borda.boundary.plot(ax=ax[0],edgecolor='black',linewidth=0.3)
    borda.boundary.plot(ax=ax[1],edgecolor='black',linewidth=0.3)
    borda.boundary.plot(ax=ax[2],edgecolor='black',linewidth=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsMERRA_'+pol['tag']+'.png', 
                format="png",bbox_inches='tight',dpi=300)
    
    
    #%%
    if pol['tag']=='O3':
        fig,ax = plt.subplots(1,3)
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(18*cm, 7*cm)
    else:
        fig,ax = plt.subplots(1,4)
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(18*cm, 7*cm)
    
    xb = [np.nanmin(xvMERRA[~np.isnan(dataBRAINinMERRA)[0,:,:]]),
          np.nanmax(xvMERRA[~np.isnan(dataBRAINinMERRA)[0,:,:]])]
    yb = [np.nanmin(yvMERRA[~np.isnan(dataBRAINinMERRA)[0,:,:]]),
          np.nanmax(yvMERRA[~np.isnan(dataBRAINinMERRA)[0,:,:]])]
    if pol['tag']!='O3':
        heatmap = ax[0].pcolor(lonBRAIN,latBRAIN,np.nanmean(dataBRAIN[:,0,:,:],axis=0)*pol['conv'],
                                vmin=np.nanmin([np.nanmin(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                                np.nanmin(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*1.2,
                                vmax=np.nanmax([np.nanmax(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                                np.nanmax(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*0.8,
                                #norm=matplotlib.colors.LogNorm(
                                #    vmin=np.nanmin([np.nanmin(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                #                    np.nanmin(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*1.2,
                                #    vmax=np.nanmax([np.nanmax(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                #                    np.nanmax(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*0.8)
                                cmap='Spectral_r')
    else:
        heatmap = ax[0].pcolor(lonBRAIN,latBRAIN,np.nanmean(dataBRAIN[:,0,:,:],axis=0)*pol['conv'],
                                cmap='Spectral_r')
    ax[0].set_xlim([xb[0], xb[1]])
    ax[0].set_ylim([yb[0], yb[1]])
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
    if pol['tag']!='O3':
        heatmap1 = ax[1].pcolor(lonMERRA,latMERRA,np.nanmean(dataBRAINinMERRA[:,:,:],axis=0),
                                vmin=np.nanmin([np.nanmin(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                                np.nanmin(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*1.2,
                                vmax=np.nanmax([np.nanmax(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                                np.nanmax(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*0.8,
                                # norm=matplotlib.colors.LogNorm(
                                #    vmin=np.nanmin([np.nanmin(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                #                    np.nanmin(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*1.2,
                                #    vmax=np.nanmax([np.nanmax(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                #                    np.nanmax(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*0.8)
                                cmap='Spectral_r')
    else:
        heatmap1 = ax[1].pcolor(lonMERRA,latMERRA,np.nanmean(dataBRAINinMERRA[:,:,:],axis=0),
                            cmap='Spectral_r')
        
    ax[1].set_xlim([xb[0], xb[1]])
    ax[1].set_ylim([yb[0], yb[1]])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    cbar = fig.colorbar(heatmap1,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',
                        #norm=matplotlib.colors.LogNorm(vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                        #                               vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]))
                        )
    #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
    cbar.ax.set_xlabel(pol['tag']+' ('+pol['Unit']+')\n b) BRAIN regrid', rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 2
    cbar.ax.tick_params(labelsize=6)
    if pol['tag']!='O3':
        heatmap = ax[2].pcolor(lonMERRA,latMERRA,np.nanmean(dataMERRAfiltered[:,:,:],axis=0),
                                vmin=np.nanmin([np.nanmin(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                                np.nanmin(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*1.2,
                                vmax=np.nanmax([np.nanmax(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                                np.nanmax(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*0.8,
                                # norm=matplotlib.colors.LogNorm(
                                #    vmin=np.nanmin([np.nanmin(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                #                    np.nanmin(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*1.2,
                                #    vmax=np.nanmax([np.nanmax(np.nanmean(dataMERRAfiltered[:,:,:],axis=0)),
                                #                    np.nanmax(np.nanmean(dataBRAINinMERRA[:,:,:],axis=0))])*0.8)
                                cmap='Spectral_r')
        ax[2].set_xlim([xb[0], xb[1]])
        ax[2].set_ylim([yb[0], yb[1]])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                            #extend='both', 
                            #ticks=bounds,
                            #spacing='uniform',
                            orientation='horizontal',
                            norm=matplotlib.colors.LogNorm(vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                                            vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]))
                            )
        #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
        cbar.ax.set_xlabel(pol['tag']+' ('+pol['Unit']+')\n c) MERRA2', rotation=0,fontsize=6)
        cbar.ax.get_xaxis().labelpad = 2
        cbar.ax.tick_params(labelsize=6)
    else:
        heatmap = ax[2].pcolor(lonMERRA,latMERRA,np.nanmean(dataMERRAfiltered[:,:,:],axis=0),
                        cmap='Spectral_r')
        ax[2].set_xlim([xb[0], xb[1]])
        ax[2].set_ylim([yb[0], yb[1]])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                            #extend='both', 
                            #ticks=bounds,
                            #spacing='uniform',
                            orientation='horizontal',
                            norm=matplotlib.colors.LogNorm(vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                                                            vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]))
                            )
        #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
        cbar.ax.set_xlabel(pol['tag']+' (Dobsons)\n c) MERRA2', rotation=0,fontsize=6)
        cbar.ax.get_xaxis().labelpad = 2
        cbar.ax.tick_params(labelsize=6)
        
    if pol['tag']!='O3':
        heatmap = ax[3].pcolor(lonMERRA,latMERRA,
                      np.nanmean(dataMERRAfiltered,axis=0)-np.nanmean(dataBRAINinMERRA,axis=0),
                      vmin=-np.nanmax(np.nanmean(dataMERRAfiltered,axis=0)-np.nanmean(dataBRAINinMERRA,axis=0))*0.9,
                      vmax=np.nanmax(np.nanmean(dataMERRAfiltered,axis=0)-np.nanmean(dataBRAINinMERRA,axis=0))*0.9,
                      #vmin=np.nanmin(np.nanmean(dataMERRAfiltered-dataBRAINinMERRA)),
                      #vmax=abs(np.nanmin(np.nanmean(dataMERRAfiltered-dataBRAINinMERRA))),
                      cmap='RdBu_r')
        ax[3].set_xlim([xb[0], xb[1]])
        ax[3].set_ylim([yb[0], yb[1]])
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        borda.boundary.plot(ax=ax[3],edgecolor='black',linewidth=0.3)
        cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                            #extend='both', 
                            #ticks=bounds,
                            #spacing='uniform',
                            orientation='horizontal',
                            #norm=matplotlib.colors.LogNorm()
                            )
        #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
        cbar.ax.set_xlabel(pol['tag']+' ('+pol['Unit']+')\n d) MERRA2 - BRAIN regrid', rotation=0,fontsize=6)
        cbar.ax.get_xaxis().labelpad = 2
        cbar.ax.tick_params(labelsize=6)
        
    borda.boundary.plot(ax=ax[0],edgecolor='black',linewidth=0.3)
    borda.boundary.plot(ax=ax[1],edgecolor='black',linewidth=0.3)
    borda.boundary.plot(ax=ax[2],edgecolor='black',linewidth=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsMERRA_average_'+pol['tag']+'.png', 
                format="png",bbox_inches='tight',dpi=300)
    
    
    #%%
    shape_path= '/media/leohoinaski/HDD/shapefiles/BR_regions.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
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
    
    for sigla in dataShp['NM_MUN'].values:
        uf = dataShp[dataShp['NM_MUN']==sigla]
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
 
        ax.annotate('a) '+sigla+'\nρ = {:.2f}'.format(corr),
                xy=(0.57, 0.1), xycoords='axes fraction', 
                fontsize=8, ha='left', va='center')
        
        y,preds = xy[0,:],xy[1,:]
        
        
        if (pol['tag']=='CO') or (pol['tag']=='SO2') or (pol['tag']=='PM25'):
            ax.plot([np.nanmin([y,preds]), np.nanmax([y,preds])],
                      [np.nanmin([y,preds]), np.nanmax([y,preds])], 'k-', lw=1,dashes=[2, 2])
            ax.fill_between(np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataMERRAfiltered.shape[0]), 
                            np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataBRAINinMERRA.shape[0])*0.5,
                            alpha=0.2,facecolor='gray',edgecolor=None)
            ax.fill_between(np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataMERRAfiltered.shape[0]),
                            np.linspace(np.nanmax([y,preds]),np.nanmax([y,preds]),dataMERRAfiltered.shape[0]),
                            np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataBRAINinMERRA.shape[0],dataMERRAfiltered.shape[0])*2,
                            alpha=0.2,facecolor='gray',edgecolor=None)
            ax.set_xlim([np.nanmin([y,preds]),np.nanmax([y,preds])])
            ax.set_ylim([np.nanmin([y,preds]),np.nanmax([y,preds])])
            ax.set_aspect('equal')
            ax.set_xlabel('MERRA2\n'+pol['tag']+ ' ('+pol['Unit']+')',fontsize=8)
            ax.set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            
        else:
            ax.set_xlim([np.nanmin([y]),np.nanmax([y])])
            ax.set_ylim([np.nanmin([preds]),np.nanmax([preds])])
            ax.set_xlabel('MERRA2\n'+pol['tag']+ ' (Dobsons)',fontsize=8)
            ax.set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            
            
        fig.tight_layout()
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsMERRA_scatter_'+sigla+'_'+pol['tag']+'.png', 
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

    ax.annotate('a) '+sigla+'\nρ = {:.2f}'.format(corr),
            xy=(0.57, 0.1), xycoords='axes fraction', 
            fontsize=8, ha='left', va='center')
    
    y,preds = xy[0,:],xy[1,:]

    
    if (pol['tag']=='CO') or (pol['tag']=='SO2') or (pol['tag']=='PM25'):
        ax.plot([np.nanmin([y,preds]), np.nanmax([y,preds])],
                  [np.nanmin([y,preds]), np.nanmax([y,preds])], 'k-', lw=1,dashes=[2, 2])
        ax.fill_between(np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataMERRAfiltered.shape[0]), 
                        np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataBRAINinMERRA.shape[0])*0.5,
                        alpha=0.2,facecolor='gray',edgecolor=None)
        ax.fill_between(np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataMERRAfiltered.shape[0]),
                        np.linspace(np.nanmax([y,preds]),np.nanmax([y,preds]),dataMERRAfiltered.shape[0]),
                        np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataBRAINinMERRA.shape[0],dataMERRAfiltered.shape[0])*2,
                        alpha=0.2,facecolor='gray',edgecolor=None)
        ax.set_xlim([np.nanmin([y,preds]),np.nanmax([y,preds])])
        ax.set_ylim([np.nanmin([y,preds]),np.nanmax([y,preds])])
        ax.set_aspect('equal')
        ax.set_xlabel('MERRA2\n'+pol['tag']+ ' ('+pol['Unit']+')',fontsize=8)
        ax.set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
    else:
        ax.set_xlim([np.nanmin([y]),np.nanmax([y])])
        ax.set_ylim([np.nanmin([preds]),np.nanmax([preds])])
        ax.set_xlabel('MERRA2\n'+pol['tag']+ ' (Dobsons)',fontsize=8)
        ax.set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsMERRA_scatter_'+sigla+'_'+pol['tag']+'.png', 
                format="png",bbox_inches='tight',dpi=300)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    
    #%%
    shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
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
    
    for sigla in dataShp['NM_PAIS'].values:
        uf = dataShp[dataShp['NM_PAIS']==sigla]
        xlon,ylat = xvMERRA,yvMERRA
        
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
 
        ax.annotate('a) '+sigla+'\nρ = {:.2f}'.format(corr),
                xy=(0.57, 0.1), xycoords='axes fraction', 
                fontsize=8, ha='left', va='center')
        
        y,preds = xy[0,:],xy[1,:]

        
        
        if (pol['tag']=='CO') or (pol['tag']=='SO2') or (pol['tag']=='PM25'):
            ax.plot([np.nanmin([y,preds]), np.nanmax([y,preds])],
                      [np.nanmin([y,preds]), np.nanmax([y,preds])], 'k-', lw=1,dashes=[2, 2])
            ax.fill_between(np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataMERRAfiltered.shape[0]), 
                            np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataBRAINinMERRA.shape[0])*0.5,
                            alpha=0.2,facecolor='gray',edgecolor=None)
            ax.fill_between(np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataMERRAfiltered.shape[0]),
                            np.linspace(np.nanmax([y,preds]),np.nanmax([y,preds]),dataMERRAfiltered.shape[0]),
                            np.linspace(np.nanmin([y,preds]), np.nanmax([y,preds]),dataBRAINinMERRA.shape[0],dataMERRAfiltered.shape[0])*2,
                            alpha=0.2,facecolor='gray',edgecolor=None)
            ax.set_xlim([np.nanmin([y,preds]),np.nanmax([y,preds])])
            ax.set_ylim([np.nanmin([y,preds]),np.nanmax([y,preds])])
            ax.set_xlabel('MERRA2\n'+pol['tag']+ ' ('+pol['Unit']+')',fontsize=8)
            ax.set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.set_aspect('equal')
        else:
            print('nofact2')
            ax.set_ylim([np.nanmin([preds]),np.nanmax([preds])])
            ax.set_xlim([np.nanmin([y]),np.nanmax([y])])
            ax.set_xlabel('MERRA2\n'+pol['tag']+ ' (Dobsons)',fontsize=8)
            ax.set_ylabel('BRAIN\n'+pol['tag'] + ' ('+pol['Unit']+')',fontsize=8)
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            
        fig.tight_layout()
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsMERRA_scatter_'+sigla+'_'+pol['tag']+'.png', 
                    format="png",bbox_inches='tight',dpi=300)
#%%  city figures 

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
    aveData = dataBRAINinMERRA
    aveData2 = dataMERRAfiltered
    datesTime = datesTimeBRAIN
    xlon,ylat =xvMERRA,yvMERRA 
    
    #cmap = 'YlOrRd'
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["azure","lightgray","gold","orange","red"])

    # legend = 'BRAIN ' +pol['Pollutant'] +' ('+ pol['Unit'] + ')'
    # #legend ='BRAIN'
    # for IBGE_CODE in capitals.IBGE_CODE:
    #     IBGE_CODE=str(IBGE_CODE)
    #     s,cityMat,cityBuffer=citiesBufferINdomain(xlon,ylat,cities,IBGE_CODE)
    #     #IBGE_CODE=1100205 #    
    #     cityData,cityDataPoints,cityDataFrame,matData= dataINcity(aveData,datesTime,cityMat,s,IBGE_CODE)
    #     cityTimeSeries(cityDataFrame,matData,cities,IBGE_CODE,cmap,legend,
    #                         xlon,ylat,None,
    #                         os.path.dirname(BASE)+'/figures/',pol['tag'],'BRAIN_'+str(IBGE_CODE))
    
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["azure","lightgray","pink","deeppink","purple"])
    legend = 'MERRA2 ' +pol['Pollutant'] +' ('+ pol['Unit'] + ')'
    #legend ='BRAIN'
    for IBGE_CODE in capitals.IBGE_CODE:
        IBGE_CODE=str(IBGE_CODE)
        s,cityMat,cityBuffer=citiesBufferINdomain(xlon,ylat,cities,IBGE_CODE)
        #IBGE_CODE=1100205 #    
        cityData,cityDataPoints,cityDataFrame,matData= dataINcity(aveData2,datesTime,cityMat,s,IBGE_CODE)
        cityTimeSeries(cityDataFrame,matData,cities,IBGE_CODE,cmap,legend,
                            xlon,ylat,None,
                            os.path.dirname(BASE)+'/figures/',pol['tag'],'MERRA2_'+str(IBGE_CODE))
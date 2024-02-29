#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:03:00 2024

@author: leohoinaski
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
#%%
def brainPcolor(BASE,pol,lonBRAIN,latBRAIN,dataBRAIN,
                lonMERRA,latMERRA,dataMERRA,borda):

    fig,ax = plt.subplots(1,2)
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(18*cm, 7*cm)
    
    

    heatmap = ax[0].pcolor(lonBRAIN,latBRAIN,dataBRAIN*pol['conv'],
                            #vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                            #vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])]),
                            #norm=matplotlib.colors.LogNorm(
                            #    vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])])*1.2,
                            #    vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])])*0.8),
                            cmap='Spectral_r')
        

    ax[0].set_xlim([lonBRAIN.min(), lonBRAIN.max()])
    ax[0].set_ylim([latBRAIN.min(), latBRAIN.max()])
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
                            #norm=matplotlib.colors.LogNorm(
                            #    vmin=np.nanmin([np.nanmin(dataMERRAfiltered[:,:,:]),np.nanmin(dataBRAINinMERRA[:,:,:])]),
                            #    vmax=np.nanmax([np.nanmax(dataMERRAfiltered[:,:,:]),np.nanmax(dataBRAINinMERRA[:,:,:])])),
                            cmap='Spectral_r')
    ax[1].set_xlim([lonBRAIN.min(), lonBRAIN.max()])
    ax[1].set_ylim([latBRAIN.min(), latBRAIN.max()])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                        #extend='both', 
                        #ticks=bounds,
                        #spacing='uniform',
                        orientation='horizontal',
                        norm=matplotlib.colors.LogNorm(vmin=np.nanmin(dataMERRA),
                                                        vmax=np.nanmax(dataMERRA)),
                        )
    #cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
    cbar.ax.set_xlabel(pol['tag']+' tropospheric column $\mol.m^{-2}$\n b) SENTINEL/TROPOMI', rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 2
    cbar.ax.tick_params(labelsize=6)
 
    
    borda.boundary.plot(ax=ax[0],edgecolor='black',linewidth=0.3)
    borda.boundary.plot(ax=ax[1],edgecolor='black',linewidth=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsSENTINEL_average_'+pol['tag']+'.png', 
                format="png",bbox_inches='tight')
    
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
    
    #%%
import geopandas as gpd
import scipy
def BRAINscattersRegions(shape_path,BASE,pol,xvMERRA,yvMERRA,dataMERRAfiltered,dataBRAINinMERRA):
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_regions.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
    dataShp = gpd.read_file(shape_path)
        
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
        fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsSENTINEL_scatter_'+sigla+'_'+pol['tag']+'.png', 
                    format="png",bbox_inches='tight')
        
        
    
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
    fig.savefig(os.path.dirname(BASE)+'/figures/BRAINvsSENTINEL_scatter_'+sigla+'_'+pol['tag']+'.png', 
                format="png",bbox_inches='tight')
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
                    format="png",bbox_inches='tight')
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
                   bbox_inches='tight')
        return matData.shape


def timeseriesSelection(BASE,datesTimeBRAIN,meanEvents,aveMeanEvents,pol,domain):
    fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [6, 1],'wspace':0, 'hspace':0},
                           sharey=True)
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(18*cm, 9*cm)
    # Time series das médias no domínio
    ax[0].plot(datesTimeBRAIN['datetime'],meanEvents,linewidth=0.5,label='Average',
               c='red',zorder=0)
    #ax.set_yscale('log')
    ax[0].set_ylabel('Air quality \n'+pol['tag']+ ' ('+pol['Unit']+')' ,fontsize=8)
    ax[0].set_xlabel(None ,fontsize=8)
    ax[0].xaxis.set_tick_params(labelsize=7,rotation=30)
    ax[0].yaxis.set_tick_params(labelsize=7)
    ax[0].set_xlim([np.nanmin(datesTimeBRAIN['datetime']),np.nanmax(datesTimeBRAIN['datetime'])])
    ax[0].fill_between(datesTimeBRAIN['datetime'],np.nanmin(meanEvents), aveMeanEvents, alpha=0.5,color='white')
    # Boxplot da média dos domínios
    #ax[1].boxplot(meanEvents)
    ax[1].hist(meanEvents, orientation = "horizontal",color='red',edgecolor = "black", alpha=0.7)
    #ax[1].set_xticks([])
    ax[1].set_xlabel('Events' ,fontsize=8)
    ax[1].yaxis.set_tick_params(labelsize=7)
    ax[1].xaxis.set_tick_params(labelsize=7)
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/timeSeriesSelection_'+domain+'_'+
                str(pol['Criteria'])+'_'+pol['tag']+'.png', format="png",
               bbox_inches='tight',dpi=300)
    return fig
    # capitals = pd.read_csv(os.path.dirname(BASE)+'/data/BR_capitais.csv')  
    
    # shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Municipios_2020.shp'
    
    # cities = gpd.read_file(shape_path)
    # cities.crs = "EPSG:4326"
    # aveData = dataBRAINinMERRA
    # aveData2 = dataMERRAfiltered
    # datesTime = datesTimeBRAIN
    # xlon,ylat =xvMERRA,yvMERRA 
    
    # #cmap = 'YlOrRd'
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
    
    
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["azure","lightgray","pink","deeppink","purple"])
    # legend = 'MERRA2 ' +pol['Pollutant'] +' ('+ pol['Unit'] + ')'
    # #legend ='BRAIN'
    # for IBGE_CODE in capitals.IBGE_CODE:
    #     IBGE_CODE=str(IBGE_CODE)
    #     s,cityMat,cityBuffer=citiesBufferINdomain(xlon,ylat,cities,IBGE_CODE)
    #     #IBGE_CODE=1100205 #    
    #     cityData,cityDataPoints,cityDataFrame,matData= dataINcity(aveData2,datesTime,cityMat,s,IBGE_CODE)
    #     cityTimeSeries(cityDataFrame,matData,cities,IBGE_CODE,cmap,legend,
    #                         xlon,ylat,None,
    #                         os.path.dirname(BASE)+'/figures/',pol['tag'],'MERRA2_'+str(IBGE_CODE))
    
def exceedingStats(BASE,dataBox,dataShp,pol,polEmis,ds1,dataBoxAQ,dataBoxPixel,domain):
    fig,ax = plt.subplots(3,1,sharex=True,gridspec_kw={'wspace':0, 'hspace':0.05})
    nCriticos = []
    for ll in dataBox:
        if len(ll)>0:
            nCriticos.append(np.nanmax(ll))
        else:
            nCriticos.append(0)
    bplot1 = ax[0].boxplot(dataBox,
               notch=True,  # notch shape
               vert=True,  # vertical box alignment
               patch_artist=True,
               flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white','alpha':0.5}
               )
    try:
        #ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        ticks = [i+1 for i, v in enumerate(dataShp['NM_MESO'])]
        #ax[0].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[0].set_xticks(ticks, dataShp['NM_MESO'],fontsize=7)
        ax[0].tick_params(axis='both', which='major', labelsize=6)
        ax[0].set_ylabel(polEmis+' emission\n'+'('+ds1[polEmis].units.split(' ')[0]+')',fontsize=8)
        ax[0].set_yscale('symlog')
        ax[0].set_ylim([0,np.max(nCriticos)])
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(18*cm, 12*cm)
        # fill with colors
        #colors = np.repeat(['#E72C31'],dataShp['UF'].shape[0])
        colors = np.repeat(['#E72C31'],dataShp['NM_MESO'].shape[0])
    except:
        #ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        #ax[0].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[0].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[0].tick_params(axis='both', which='major', labelsize=6)
        ax[0].set_ylabel(polEmis+' emission\n'+'('+ds1[polEmis].units.split(' ')[0]+')',fontsize=8)
        ax[0].set_yscale('symlog')
        ax[0].set_ylim([0,np.max(nCriticos)])
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(18*cm, 12*cm)
        # fill with colors
        #colors = np.repeat(['#E72C31'],dataShp['UF'].shape[0])
        colors = np.repeat(['#E72C31'],dataShp['UF'].shape[0])
    
    
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot1['medians']:
        median.set_color('black')
        
    nCriticos = []
    for ll in dataBoxAQ:
        nCriticos.append(len(ll))

    try:
        ticks = [i+1 for i, v in enumerate(dataShp['NM_MESO'])]
        bplot1 = ax[1].bar(ticks,nCriticos,color='#E72C31')
        #ax[1].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[1].set_xticks(ticks, dataShp['NM_MESO'],fontsize=7)
        ax[1].tick_params(axis='both', which='major', labelsize=6)
        ax[1].set_ylabel('Exceeding events\n'+polEmis,fontsize=8)
        ax[1].set_yscale('log')
        
        #ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        ticks = [i+1 for i, v in enumerate(dataShp['NM_MESO'])]
        bplot1 = ax[2].bar(ticks,dataBoxPixel,color='#E72C31')
        #ax[2].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[1].set_xticks(ticks, dataShp['NM_MESO'],fontsize=7)
    except:
        ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        bplot1 = ax[1].bar(ticks,nCriticos,color='#E72C31')
        #ax[1].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[1].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[1].tick_params(axis='both', which='major', labelsize=6)
        ax[1].set_ylabel('Exceeding events\n'+polEmis,fontsize=8)
        ax[1].set_yscale('log')
        
        #ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        bplot1 = ax[2].bar(ticks,dataBoxPixel,color='#E72C31')
        #ax[2].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[1].set_xticks(ticks, dataShp['UF'],fontsize=7)
        
    ax[2].tick_params(axis='both', which='major', labelsize=6)
    ax[2].set_ylabel('Exceeding pixels\n'+polEmis,fontsize=8)
    ax[2].set_yscale('log')
    
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/boxplotViolateEmissions_'+
                pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
               bbox_inches='tight',dpi=300)
    return fig

def Qscatter(BASE,q1EMIS,q1BRAIN,q2EMIS,q2BRAIN,q3EMIS,q3BRAIN,q4EMIS,q4BRAIN,
             pol,polEmis,minMeanEmis,domain):
    # FIGURA SCATTER DOS QUADRANTES
    fig, ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(12*cm, 12*cm)
    q4 = ax.scatter(q4EMIS,q4BRAIN,
               s=.1,alpha=1,c='#E72C31', label = 'Q4')
        
    q1 = ax.scatter(q1EMIS,q1BRAIN,
               s=.1,alpha=1,c='#71CCF1', label = 'Q1')
    
    q2 = ax.scatter(q3EMIS,q3BRAIN,
               s=.1,alpha=1,c='#FDC45C', label = 'Q3')
    
    q3 = ax.scatter(q2EMIS,q2BRAIN,
               s=.1,alpha=1,c='#FF7533', label = 'Q2')
    
    if pol['Criteria']!=None:
        ax.axhline(y=pol['Criteria'], color='red', linestyle='--',linewidth=1,
                      label='Air quality standard')
        ax.axvline(x=minMeanEmis, 
                   color='gray', linestyle='--',linewidth=1,
                       label='Average of significant emissions')
    
    #ax.legend(loc='lower left' ,fontsize=8, markerscale=15, frameon=False)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,3,2,0,4,5]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
              loc='lower left' ,fontsize=8, markerscale=15, frameon=False)

    
    q1.set_alpha(0.2)
    q2.set_alpha(0.2)
    q3.set_alpha(0.2)
    q4.set_alpha(0.2)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Air quality\n'+pol['tag']+ ' ('+pol['Unit']+')',fontsize=8)
    ax.set_xlabel('Emission\n'+polEmis ,fontsize=8)
        
        # You can also use lh.set_sizes([50])
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/scatterQ_'+domain+'_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
               bbox_inches='tight',dpi=300)
    
def QscatterAll(BASE,q1EMIS,q1BRAIN,q2EMIS,q2BRAIN,q3EMIS,q3BRAIN,q4EMIS,q4BRAIN,
             pol,polEmis,minMeanEmis,dataBRAIN,dataEMIS,domain):
    
    ###calculate Spearman correlation using new_df
    try:
        nas = np.logical_or(np.isnan(dataBRAIN.data.flatten()), np.isnan(dataEMIS.data.flatten()))
    except:
         nas = np.logical_or(np.isnan(dataBRAIN.flatten()), np.isnan(dataEMIS.flatten()))
   
    corr, p_value = scipy.stats.spearmanr(dataBRAIN.flatten()[~nas],dataEMIS.flatten()[~nas])
   
    ###insert text with Spearman correlation
    # ax.annotate('ρ = {:.2f}'.format(corr), 
    #         xy=(0.70, 0.9), xycoords='axes fraction', 
    #         fontsize=8, ha='left', va='center')
 

    
    # FIGURA SCATTER DOS QUADRANTES
    fig, ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(12*cm, 12*cm)
    q4 = ax.scatter(q4EMIS,q4BRAIN,
               s=.1,alpha=1,c='#E72C31', label = None)
        
    q1 = ax.scatter(q1EMIS,q1BRAIN,
               s=.1,alpha=1,c='#71CCF1', label = None)
    
    q2 = ax.scatter(q3EMIS,q3BRAIN,
               s=.1,alpha=1,c='#71CCF1', label = None)
    
    q3 = ax.scatter(q2EMIS,q2BRAIN,
               s=.1,alpha=1,c='#E72C31', label = None)
    
    if pol['Criteria']!=None:
        ax.axhline(y=pol['Criteria'], color='red', linestyle='--',linewidth=1,
                      label='Air quality standard')
        # ax.axvline(x=minMeanEmis, 
        #            color='gray', linestyle='--',linewidth=1,
        #                label='Average of significant emissions')
    
    ax.legend(loc='lower left' ,fontsize=8, markerscale=15, frameon=False)
    handles, labels = plt.gca().get_legend_handles_labels()
    # order = [1,3,2,0,4]
    # ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
    #           loc='lower left' ,fontsize=8, markerscale=15, frameon=False)
   
    q1.set_alpha(0.2)
    q2.set_alpha(0.2)
    q3.set_alpha(0.2)
    q4.set_alpha(0.2)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Air quality\n'+pol['tag']+ ' ('+pol['Unit']+')',fontsize=8)
    ax.set_xlabel('Emission\n'+polEmis ,fontsize=8)
        
        # You can also use lh.set_sizes([50])
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/scatterQALLAQS_'+domain+'_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
               bbox_inches='tight',dpi=300)
    
    # FIGURA SCATTER DOS QUADRANTES
    fig, ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(12*cm, 12*cm)
    q4 = ax.scatter(q4EMIS,q4BRAIN,
               s=.1,alpha=1,c='#71CCF1', label = None)
        
    q1 = ax.scatter(q1EMIS,q1BRAIN,
               s=.1,alpha=1,c='#71CCF1', label = None)
    
    q2 = ax.scatter(q3EMIS,q3BRAIN,
               s=.1,alpha=1,c='#71CCF1', label = None)
    
    q3 = ax.scatter(q2EMIS,q2BRAIN,
               s=.1,alpha=1,c='#71CCF1', label = None)
    
    # if pol['Criteria']!=None:
    #     ax.axhline(y=pol['Criteria'], color='red', linestyle='--',linewidth=1,
    #                   label='Air quality standard')
    #     ax.axvline(x=minMeanEmis, 
    #                color='gray', linestyle='--',linewidth=1,
    #                    label='Average of significant emissions')
    
    #ax.legend(loc='lower left' ,fontsize=8, markerscale=15, frameon=False)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,3,2,0]
    # ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
    #           loc='lower left' ,fontsize=8, markerscale=15, frameon=False)

    ax.annotate('ρ = {:.2f}'.format(corr),
            xy=(0.67, 0.1), xycoords='axes fraction', 
            fontsize=8, ha='left', va='center')
    
    q1.set_alpha(0.2)
    q2.set_alpha(0.2)
    q3.set_alpha(0.2)
    q4.set_alpha(0.2)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Air quality\n'+pol['tag']+ ' ('+pol['Unit']+')',fontsize=8)
    ax.set_xlabel('Emission\n'+polEmis ,fontsize=8)
        
        # You can also use lh.set_sizes([50])
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/scatterQALLraw_'+domain+'_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
               bbox_inches='tight',dpi=300)
    
def Qspatial(BASE,rootFolder,lonBRAIN,latBRAIN,freQ1,freQ2,freQ3,freQ4,pol,
             dataShp,domain):
    # Figure frequencia no Q4 - QUADRANTES NO ESPAÇO
    fig,ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    if domain=='SC':
        fig.set_size_inches(20*cm, 10*cm)
    else:
        fig.set_size_inches(10*cm, 10*cm)
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [[0,0,0,0],'#71CCF1','#71CCF1'])
    heatmap = ax.pcolor(lonBRAIN,latBRAIN,freQ1[0,:,:],cmap=cmap)
    #heatmap = ax.pcolor(lonBRAIN,latBRAIN,q1EMISmat[0,0,:,:].data,cmap=cmap)
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [[0,0,0,0],'#FF7400','#FF7400'])
    heatmap = ax.pcolor(lonBRAIN,latBRAIN,freQ2[0,:,:],cmap=cmap)
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [[0,0,0,0],'#FDC45C','#FDC45C'])
    heatmap = ax.pcolor(lonBRAIN,latBRAIN,freQ3[0,:,:],cmap=cmap)
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [[0,0,0,0],'#E72C39','#E72C31'])
    heatmap = ax.pcolor(lonBRAIN,latBRAIN,freQ4[0,:,:],cmap=cmap)
    
    
    #cbar = fig.colorbar(heatmap,fraction=0.04, pad=0.02,
                        # #ticks=bounds,
                        # #extend='both',
                        # spacing='uniform',
                        # orientation='horizontal',
                        # #norm=norm,
                        # ax=ax)
    
    #cbar.ax.tick_params(rotation=30)
    ax.set_xlim([np.nanmin(lonBRAIN), np.nanmax(lonBRAIN)])
    ax.set_ylim([np.nanmin(latBRAIN), np.nanmax(latBRAIN)]) 
    ax.set_xticks([])
    ax.set_yticks([])
    
    #shape_path= rootFolder+'/shapefiles/Brasil.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
    #dataShp = gpd.read_file(shape_path)
    dataShp.boundary.plot(ax=ax,edgecolor='black',linewidth=0.3)
    ax.set_frame_on(False)
    del heatmap
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/spatialQ_'+domain+'_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
               bbox_inches='tight',dpi=300)
    return fig

def reductionQ4(BASE,rootFolder,lonBRAIN,latBRAIN,q4EMISmatE1,polEmis,pol,
                dataBox,dataShp,domain):
    # FIGURA ABATIMENTO DAS EMISSÕES NO Q4 - ETAPA 1
    #fig,ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [5, 1],'wspace':0, 'hspace':0.4})
    fig,ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    if domain=='SC':
        fig.set_size_inches(20*cm, 10*cm)
    else:
        fig.set_size_inches(10*cm, 10*cm)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white','#FDC45C','#FF7533','#E72C31',])
    
    bounds = np.array([0,1,5,10,30,60,90,95,99,100])
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    heatmap = ax.pcolor(lonBRAIN,latBRAIN,np.nanmean(q4EMISmatE1[:,0,:,:],axis=0),cmap='rainbow',norm=norm)
    cbar = fig.colorbar(heatmap,fraction=0.03, pad=0.02,shrink=0.5,
                        ticks=bounds,
                        #extend='both',
                        spacing='uniform',
                        orientation='horizontal',
                        norm=norm,
                        ax=ax)
    cbar.ax.tick_params(rotation=30)
    cbar.ax.set_xlabel(polEmis+'\nRedução média da emissão (%)', rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 6
    cbar.ax.tick_params(labelsize=7) 
    ax.set_xlim([np.nanmin(lonBRAIN), np.nanmax(lonBRAIN)])
    ax.set_ylim([np.nanmin(latBRAIN), np.nanmax(latBRAIN)]) 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    #shape_path= rootFolder+'/shapefiles/Brasil.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
    #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
    #dataShp = gpd.read_file(shape_path)
    dataShp.boundary.plot(ax=ax,edgecolor='black',linewidth=0.3)
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/ReductionSpatial_'+domain+'_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
               bbox_inches='tight',dpi=300)
    
    fig,ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(18*cm, 4*cm)
    bplot1 = ax.boxplot(dataBox,
               notch=True,  # notch shape
               vert=True,  # vertical box alignment
               patch_artist=True,
               flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'white','alpha':0.5})
    try:
        ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        ax.set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_ylabel(polEmis+'\nRedução emissão \n(%)' ,fontsize=8)
        
        # fill with colors
        colors = np.repeat(['#E72C31'],dataShp['UF'].shape[0])
    except:
        ticks = [i+1 for i, v in enumerate(dataShp['NM_MESO'])]
        ax.set_xticks(ticks, dataShp['NM_MESO'],fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_ylabel(polEmis+'\nRedução emissão \n(%)' ,fontsize=8)
        
        # fill with colors
        colors = np.repeat(['#E72C31'],dataShp['NM_MESO'].shape[0])

    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/ReductionBox_'+domain+'_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
               bbox_inches='tight',dpi=300)
    return fig

def timeAverageFig(data,xlon,ylat,legend,cmap,borderShape,folder,pol,aveTime,domain):
    fig, ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    if domain=='SC':
        fig.set_size_inches(20*cm, 14*cm)
    else:
        fig.set_size_inches(18*cm, 13*cm)
    #cmap = plt.get_cmap(cmap, 6)
    bounds = np.sort(np.unique(np.array([np.nanpercentile(data[data>0],1),
                       np.nanpercentile(data[data>0],5),
                       np.nanpercentile(data[data>0],10),
                        np.nanpercentile(data[data>0],25),
                        np.nanpercentile(data[data>0],50),
                        np.nanpercentile(data[data>0],75),
                        np.nanpercentile(data[data>0],90),
                        np.nanpercentile(data[data>0],95),
                        np.nanpercentile(data[data>0],99),
                        np.nanpercentile(data[data>0],100)])))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    heatmap = ax.pcolor(xlon,ylat,data,cmap=cmap,norm=norm)
    cbar = fig.colorbar(heatmap,fraction=0.03, pad=0.02, shrink=0.5,
                        #extend='both',
                        ticks=bounds,
                        spacing='uniform',
                        orientation='horizontal',
                        norm=norm,
                        ax=ax)

    cbar.ax.tick_params(rotation=30)
    #tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    #cbar.locator = tick_locator
    #cbar.ax.set_xscale('log')
    #cbar.update_ticks()
    
    cbar.ax.set_xlabel(legend, rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 2
    cbar.ax.tick_params(labelsize=6)
    #cbar.ax.locator_params(axis='both',nbins=5)
    cbar.ax.minorticks_off()
    #br = gpd.read_file(borderShape)
    #br.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
    borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
    ax.set_xlim([np.nanmin(xlon), np.nanmax(xlon)])
    ax.set_ylim([np.nanmin(ylat), np.nanmax(ylat)]) 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(folder+'/timeAverageFig_'+domain+'_'+pol+'_'+aveTime+'.png',
                format="png",bbox_inches='tight')
    return fig


def exceedanceFig(data,xlon,ylat,legend,cmap,borderShape,folder,pol,aveTime,
                  domain, criteria):
    fig, ax = plt.subplots()
    cm = 1/2.54  # centimeters in inches
    if domain=='SC':
        fig.set_size_inches(20*cm, 14*cm)
    else:
        fig.set_size_inches(18*cm, 13*cm)
    #cmap = plt.get_cmap(cmap, 4)
    # cmap.set_under('white')
    # cmap.set_over('red')
    cmap.set_bad('white',1.)
    #bounds = np.concatenate((np.array([0,1]),np.linspace(2, np.nanmax([np.nanmax(data),8]), 5,dtype=int)))
    bounds = np.sort(np.unique(np.array([0,1,np.nanpercentile(data[data>0],1),
                   np.nanpercentile(data[data>0],5),
                   np.nanpercentile(data[data>0],10),
                    np.nanpercentile(data[data>0],25),
                    np.nanpercentile(data[data>0],50),
                    np.nanpercentile(data[data>0],75),
                    np.nanpercentile(data[data>0],90),
                    np.nanpercentile(data[data>0],95),
                    np.nanpercentile(data[data>0],99),
                    np.nanpercentile(data[data>0],100)])))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    #cmap.set_under('white')
    heatmap = ax.pcolor(xlon,ylat,data,cmap=cmap,norm=norm)
    #form = numFormat(data)
    cbar = fig.colorbar(heatmap,fraction=0.03, pad=0.02,format="%.0f",shrink=0.5,
                        #extend='both', 
                        ticks=bounds,
                        spacing='uniform',
                        orientation='horizontal',
                        norm=norm)
    cbar.ax.set_xticklabels(['{:.0f}'.format(x) for x in bounds],rotation=30)
    cbar.ax.set_xlabel(legend, rotation=0,fontsize=6)
    cbar.ax.get_xaxis().labelpad = 5
    cbar.ax.tick_params(labelsize=6)
    #cbar.ax.locator_params(axis='both',nbins=5)
    #br = gpd.read_file(borderShape)
    #br.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
    borderShape.boundary.plot(edgecolor='black',linewidth=0.5,ax=ax)
    ax.set_xlim([np.nanmin(xlon), np.nanmax(xlon)])
    ax.set_ylim([np.nanmin(ylat), np.nanmax(ylat)]) 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(folder+'/exceedanceFig_'+domain+'_'+str(criteria)+'_'+pol+'_'+aveTime+'.png', 
                format="png",bbox_inches='tight')
    return fig
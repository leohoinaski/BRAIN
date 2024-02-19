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


def timeseriesSelection(BASE,datesTimeBRAIN,meanEvents,aveMeanEvents,pol):
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
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/timeSeriesSelection_'+pol['tag']+'.png', format="png",
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
    
    
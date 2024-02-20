#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:09:12 2024

Este script é utilizado para analisar a relação entre emissão local e qualidade
do ar no mesmo pixel. 



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
import temporalStatistics as tst
import BRAINfigs
# -------------------------------INPUTS----------------------------------------


NO2 = {
  "Pollutant": "$NO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1880,
  "tag":'NO2',
  #"Criteria": 260, # 260, 240, 220, 200
  "Criteria_ave": 1,
}

CO = {
  "Pollutant": "CO",
  "Unit": 'ppb',
  "conv": 1000, # Conversão de ppm para ppb
  "tag":'CO',
  "Criteria_ave": 8,
}

O3 = {
  "Pollutant": "$O_{3}$",
  "Unit": 'ppm',
  "conv": 1,
  "tag":'O3',
  "Criteria_ave": 8,
}

SO2 = {
  "Pollutant": "$SO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 2620,
  "tag":'SO2',
  "Criteria_ave": 24,
  
}

PM10 = {
  "Pollutant": "$PM_{10}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM10',
  "Criteria_ave": 24,
}

PM25 = {
  "Pollutant": "$PM_{2.5}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM25',
  "Criteria_ave": 24,
}

pollutants=[NO2,SO2,O3,PM10,PM25]
pollutants=[NO2]
emisTypes = ['BRAVES','FINN','IND2CMAQ','MEGAN']
criterias = [260,240,220,200] # NO2

criterias = [240]
#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
rootFolder = os.path.dirname(os.path.dirname(BASE))
dataFolder = os.path.dirname(BASE)+'/data'
airQualityFolder =  dataFolder+'/BRAIN'
emissFolder =  dataFolder+'/EMIS'
domain = 'BR'
year = '2020'

print('Looping for each variable')
for kk,pol in enumerate(pollutants):
    for criteria in criterias:
        pol['Criteria'] = criteria
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
        
        for ii, emisType in enumerate(emisTypes):
            fileType='BRAIN_BASEMIS_'+domain+'_2019_'+emisType+'_'+polEmis+'_'+str(year)
            prefixed = sorted([filename for filename in os.listdir(emissFolder) if filename.startswith(fileType)])
            ds1 = nc.Dataset(prefixed[0])
            if ii==0:
                dataEMIS = ds1[polEmis][0:8759,:,:,:]
            else:
                dataEMIS = dataEMIS+ds1[polEmis][0:8759,:,:,:]
                
        os.chdir(os.path.dirname(BASE))
        datesTimeEMIS, dataEMIS = BRAINutils.fixTimeBRAINemis(ds1,dataEMIS[0:8759,:,:,:])
        
       
        # ========BRAIN files============
        os.chdir(airQualityFolder)
        print(pol)
        print('Openning netCDF files')
        # Opening netCDF files
        fileType='BRAIN_BASECONC_'+domain+'_'+pol['tag']+'_'+str(year)
        prefixed = sorted([filename for filename in os.listdir(airQualityFolder) if filename.startswith(fileType)])
        ds = nc.Dataset(prefixed[0])
        # Selecting variable
        dataBRAIN = ds[pol['tag']][:]
        # Get datesTime and removing duplicates
        datesTimeBRAIN, dataBRAIN = BRAINutils.fixTimeBRAIN(ds,dataBRAIN)
        latBRAIN = ds['LAT'][:]
        lonBRAIN = ds['LON'][:]
        latBRAINflat = latBRAIN.flatten()
        lonBRAINflat = lonBRAIN.flatten()
        os.chdir(os.path.dirname(BASE)+'/scripts')
        
        lia, loc = ismember.ismember(datesTimeEMIS['datetime'].astype(str), datesTimeBRAIN['datetime'].astype(str))
        dataBRAIN = dataBRAIN[loc,:,:,:]
        datesTimeBRAIN = datesTimeBRAIN.iloc[loc,:]
        
        # Converting averaging time
        if pol['Criteria_ave']==1:
            dataBRAIN = dataBRAIN.copy()
            dataEMIS = dataEMIS.copy()
        elif pol['Criteria_ave']==8:
            # Daily-maximum 8h-moving average
            dataBRAIN = tst.movingAverage(datesTimeBRAIN,dataBRAIN,8)
            datesTimeBRAIN = datesTimeBRAIN.groupby(by=['year', 'month', 'day']).size().reset_index()
            datesTimeBRAIN['datetime']=pd.to_datetime(datesTimeBRAIN[['year', 'month', 'day']])
            dataEMIS = tst.movingAverage(datesTimeEMIS,dataEMIS,8)
            datesTimeEMIS = datesTimeEMIS.groupby(by=['year', 'month', 'day']).size().reset_index()
            datesTimeEMIS['datetime']=pd.to_datetime(datesTimeEMIS[['year', 'month', 'day']])
        elif pol['Criteria_ave']==24:
            # Daily averages
            dataBRAIN, dailyData = tst.dailyAverage(datesTimeBRAIN,dataBRAIN)
            datesTimeBRAIN = datesTimeBRAIN.groupby(by=['year', 'month', 'day']).size().reset_index()
            datesTimeBRAIN['datetime']=pd.to_datetime(datesTimeBRAIN[['year', 'month', 'day']])           
            dataEMIS, dailyData = tst.dailyAverage(datesTimeEMIS,dataEMIS)
            datesTimeEMIS = datesTimeEMIS.groupby(by=['year', 'month', 'day']).size().reset_index()
            datesTimeEMIS['datetime']=pd.to_datetime(datesTimeEMIS[['year', 'month', 'day']])           
        #% Removendo dados fora do Brasil
        
        shape_path= rootFolder+'/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
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
        
        uf = dataShp[dataShp['NM_PAIS']=='Brasil']
        s,cityMat=dataINshape(lonBRAIN,latBRAIN,uf)
        
        dataBRAIN[:,:,cityMat==0] = np.nan
        dataEMIS[:,:,cityMat==0] = np.nan
        
        # Removing 1% higher
        dataBRAIN = tst.timeseriesFiltering(dataBRAIN,99)
        #dataEMIS = tst.timeseriesFiltering(dataEMIS,99.9)
        
        # ------------Média dos eventos ao logo do ano em todo domínio-----------------  
        meanEvents = np.nanpercentile(dataBRAIN[:,0,:,:].reshape(dataBRAIN.shape[0],-1), 50,axis=1)
        
        # extraindo o percentil dos eventos 
        aveMeanEvents = np.nanpercentile(meanEvents,75)
        
        # Figura seleção da timeseries
        BRAINfigs.timeseriesSelection(BASE,datesTimeBRAIN,meanEvents*pol['conv'],aveMeanEvents*pol['conv'],pol)
        
        # Detectando os eventos acima do percentil
        boolEvents = meanEvents>aveMeanEvents
        
        # -------------Emissões que violam o padrão de qualidade do ar-----------------
        # FIltro da matriz
        violEmis = dataEMIS[boolEvents,:,:,:].flatten()[dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']]   
        # Mínima emissão
        #minEmis = np.nanmin(dataEMIS[boolEvents,:,:,:].flatten()[dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']])
        # Percentil 25 
        #min25Emis = np.nanpercentile(dataEMIS[boolEvents,:,:,:].flatten()[dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']],25)
        # Percentil 50
        minMeanEmis = np.nanpercentile(dataEMIS[boolEvents,:,:,:].flatten()[dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']],50)
        
        # Filtro dos eventos com violação
        violAirQ = dataBRAIN[boolEvents,:,:,:].flatten()[dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']] 
        violDf = pd.DataFrame()
        violDf['lat'] =  np.repeat(latBRAIN[:,:,np.newaxis],dataBRAIN.shape[0],axis=2)[:,:,boolEvents].flatten()[dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']] 
        violDf['lon'] =  np.repeat(lonBRAIN[:,:,np.newaxis],dataBRAIN.shape[0],axis=2)[:,:,boolEvents].flatten()[dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']] 
        violDf['Emission'] = violEmis
        violDf['AirQ'] = violAirQ*pol['conv']
        violDf.to_csv(os.path.dirname(BASE)+'/tables'+'/boxplotViolateEmissions_'+pol['tag']+'_'+str(pol['Criteria'])+'.csv')
                     
        # Por estado
        shape_path= rootFolder+'/shapefiles/Brasil.shp'
        #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
        #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
        dataShp = gpd.read_file(shape_path)
        dataBox=[]
        dataBoxAQ=[]
        dataBoxPixel=[]
        statDf = pd.DataFrame()
        statDf['UF']=dataShp['UF']
        statDf['MAXEMIS_'+str(pol['Criteria'])] = np.nan
        statDf['AVEEMIS_'+str(pol['Criteria'])] = np.nan
        statDf['NcriticalEvents_'+str(pol['Criteria'])] = np.nan
        statDf['NcriticalPixels_'+str(pol['Criteria'])] = np.nan
        statDf['AveReduction_'+str(pol['Criteria'])] = np.nan
        statDf['MaxReduction_'+str(pol['Criteria'])] = np.nan
        
        for ii,state in enumerate(dataShp['UF']):
            uf = dataShp[dataShp['UF']==state]
            s,cityMat=dataINshape(lonBRAIN,latBRAIN,uf)
            dataEMISuf = dataEMIS[boolEvents,:,:,:]
            dataBRAINuf = dataBRAIN[boolEvents,:,:,:]
            dataBox.append(dataEMISuf[:,:,cityMat==1].flatten()[(dataBRAINuf[:,:,cityMat==1].flatten()*pol['conv']>pol['Criteria'])])
            dataBoxAQ.append(dataBRAINuf[:,:,cityMat==1].flatten()[(dataBRAINuf[:,:,cityMat==1].flatten()*pol['conv']>pol['Criteria'])])
            dataBRAINuf[:,:,cityMat==0] = np.nan
            dataBRAINuf[dataBRAINuf*pol['conv']<pol['Criteria']] = np.nan
            dataBoxPixel.append(np.sum(~np.isnan(dataBRAINuf).all(axis=0)))
            try:
                statDf['MAXEMIS_'+str(pol['Criteria'])][ii] = np.nanmax(dataEMISuf[:,:,cityMat==1].flatten()[(dataBRAINuf[:,:,cityMat==1].flatten()*pol['conv']>pol['Criteria'])])
                statDf['AVEEMIS_'+str(pol['Criteria'])][ii] = np.percentile(dataEMISuf[:,:,cityMat==1].flatten()[(dataBRAINuf[:,:,cityMat==1].flatten()*pol['conv']>pol['Criteria'])],50)
                statDf['NcriticalEvents_'+str(pol['Criteria'])][ii] = len(dataBRAINuf[:,:,cityMat==1].flatten()[(dataBRAINuf[:,:,cityMat==1].flatten()*pol['conv']>pol['Criteria'])])
                statDf['NcriticalPixels_'+str(pol['Criteria'])][ii] = np.sum(~np.isnan(dataBRAINuf).all(axis=0))
                
            except:
                statDf['MAXEMIS_'+str(pol['Criteria'])][ii] = 0
                statDf['NcriticalEvents_'+str(pol['Criteria'])][ii] = 0
                statDf['NcriticalPixels_'+str(pol['Criteria'])][ii] = 0
                statDf['AVEEMIS_'+str(pol['Criteria'])][ii] = 0
            
        
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
                   )
        ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        ax[0].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[0].tick_params(axis='both', which='major', labelsize=6)
        ax[0].set_ylabel(polEmis+' emission\n'+'('+ds1[polEmis].units.split(' ')[0]+')',fontsize=8)
        ax[0].set_yscale('symlog')
        ax[0].set_ylim([0,np.max(nCriticos)])
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(18*cm, 12*cm)
        # fill with colors
        colors = np.repeat(['#E72C31'],dataShp['UF'].shape[0])
        for patch, color in zip(bplot1['boxes'], colors):
            patch.set_facecolor(color)
        for median in bplot1['medians']:
            median.set_color('black')
            
        nCriticos = []
        for ll in dataBoxAQ:
            nCriticos.append(len(ll))
        ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        bplot1 = ax[1].bar(ticks,nCriticos,color='#E72C31')
        ax[1].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[1].tick_params(axis='both', which='major', labelsize=6)
        ax[1].set_ylabel('Exceding events\n'+polEmis,fontsize=8)
        ax[1].set_yscale('log')
        
        ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        bplot1 = ax[2].bar(ticks,dataBoxPixel,color='#E72C31')
        ax[2].set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax[2].tick_params(axis='both', which='major', labelsize=6)
        ax[2].set_ylabel('Exceding pixels\n'+polEmis,fontsize=8)
        ax[2].set_yscale('log')
        
        fig.savefig(os.path.dirname(BASE)+'/figures'+'/boxplotViolateEmissions_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
                   bbox_inches='tight',dpi=300)
        
        #%%
        
        #% Encontrando dados em cada quadrante
        #dataBRAINflat = dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']
        #dataEMISflat = dataEMIS[boolEvents,:,:,:].flatten()
        
        # Q1 - BAIXA EMISSÃO E BOA QUALIDADE DO AR
        q1BRAIN = (dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv'])[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']<pol['Criteria']) &  (dataEMIS[boolEvents,:,:,:].flatten()<minMeanEmis)]
        q1EMIS = dataEMIS[boolEvents,:,:,:].flatten()[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']<pol['Criteria']) & (dataEMIS[boolEvents,:,:,:].flatten()<minMeanEmis)]
        
        q1EMISmat = dataEMIS[boolEvents,:,:,:]
        q1EMISmat[(dataBRAIN[boolEvents,:,:,:]*pol['conv']<pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]<minMeanEmis)]=np.nan
        #freQ1 = np.nansum(np.isnan(q1EMISmat).reshape(q1EMISmat.shape),axis=0)
        freQ1 = np.isnan(q1EMISmat).reshape(q1EMISmat.shape).all(axis=0)
        freQ1[:,cityMat==0] = False   
        
        # Q2 - BAIXA EMISSÃO E MÁ QUALIDADE DO AR
        q2BRAIN = (dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv'])[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:].flatten()<minMeanEmis)]
        q2EMIS = dataEMIS[boolEvents,:,:,:].flatten()[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:].flatten()<minMeanEmis)]
        q2EMISmat = dataEMIS[boolEvents,:,:,:]
        q2EMISmat[(dataBRAIN[boolEvents,:,:,:]*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]<minMeanEmis)]=np.nan
        #freQ2 = np.nansum(np.isnan(q2EMISmat).reshape(q2EMISmat.shape),axis=0)
        freQ2 = np.isnan(q2EMISmat).reshape(q2EMISmat.shape).any(axis=0)
        freQ2[:,cityMat==0] = False
        
        # Q3 - ALTA EMISSÃO E BOA QUALIDADE DO AR
        q3BRAIN = (dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv'])[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']<pol['Criteria']) & (dataEMIS[boolEvents,:,:,:].flatten()>minMeanEmis)]
        q3EMIS = dataEMIS[boolEvents,:,:,:].flatten()[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']<pol['Criteria']) & (dataEMIS[boolEvents,:,:,:].flatten()>minMeanEmis)]
        q3EMISmat = dataEMIS[boolEvents,:,:,:]
        q3EMISmat[(dataBRAIN[boolEvents,:,:,:]*pol['conv']<pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]>minMeanEmis)]=np.nan
        #freQ3 = np.nansum(np.isnan(q3EMISmat).reshape(q3EMISmat.shape),axis=0)
        freQ3 = np.isnan(q3EMISmat).reshape(q3EMISmat.shape).any(axis=0)
        freQ3[:,cityMat==0] = False
        
        # Q4 - ALTA EMISSÃO E MÁ QUALIDADE DO AR
        q4BRAIN = (dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv'])[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:].flatten()>minMeanEmis)]
        q4EMIS = dataEMIS[boolEvents,:,:,:].flatten()[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:].flatten()>minMeanEmis)]
        q4EMISmat = dataEMIS[boolEvents,:,:,:]
        q4EMISmat[(dataBRAIN[boolEvents,:,:,:]*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]>minMeanEmis)]=np.nan
        #freQ4 = np.nansum(np.isnan(q4EMISmat).reshape(q4EMISmat.shape),axis=0)
        freQ4 = np.isnan(q4EMISmat).reshape(q4EMISmat.shape).any(axis=0)
        freQ4[:,cityMat==0] = False
        
        # Matriz do BRAIN para o quadrante 4
        q4BRAINmat = dataBRAIN[boolEvents,:,:,:]*pol['conv']
        q4BRAINmat[(dataBRAIN[boolEvents,:,:,:]*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]>minMeanEmis)]=np.nan
        
        # EFICIENCIA DE ABATIMENTO ETAPA 1
        # Redução da emissão para os níveis do Q2
        q4EMISmat2 = dataEMIS[boolEvents,:,:,:]
        q4EMISmat2[~((dataBRAIN[boolEvents,:,:,:]*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]>minMeanEmis))]=np.nan
        q4EMISmatE1 = ((q4EMISmat2-minMeanEmis)/q4EMISmat2)*100
        
        #%%
        del dataBRAIN, dataEMIS,ds, lonBRAINflat, latBRAINflat, ds1
        del q1EMISmat,q2EMISmat,q3EMISmat,q4EMISmat,violDf,violAirQ,violEmis
        
        # FIGURA SCATTER DOS QUADRANTES
        fig, ax = plt.subplots()
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
        fig.savefig(os.path.dirname(BASE)+'/figures'+'/scatterQ_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
                   bbox_inches='tight',dpi=300)
        
        
        
        # Figure frequencia no Q4 - QUADRANTES NO ESPAÇO
        fig,ax = plt.subplots()
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [[0,0,0,0],'#71CCF1','#71CCF1'])
        heatmap = ax.pcolor(lonBRAIN,latBRAIN,freQ1[0,:,:].data,cmap=cmap)
        
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
        ax.set_xlim([lonBRAIN.min(), lonBRAIN.max()])
        ax.set_ylim([latBRAIN.min(), latBRAIN.max()])
        ax.set_xticks([])
        ax.set_yticks([])
        
        shape_path= rootFolder+'/shapefiles/BR_regions.shp'
        #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
        #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
        dataShp = gpd.read_file(shape_path)
        dataShp.boundary.plot(ax=ax,edgecolor='black',linewidth=0.3)
        ax.set_frame_on(False)
        fig.savefig(os.path.dirname(BASE)+'/figures'+'/spatialQ_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
                   bbox_inches='tight',dpi=300)
        
        del heatmap
        # FIGURA ABATIMENTO DAS EMISSÕES NO Q4 - ETAPA 1
        fig,ax = plt.subplots()
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white','#FDC45C','#FF7533','#E72C31',])
        
        bounds = np.array([0,1,5,10,30,60,90,95,99,100])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        heatmap = ax.pcolor(lonBRAIN,latBRAIN,np.nanmean(q4EMISmatE1[:,0,:,:],axis=0),cmap='rainbow',norm=norm)
        cbar = fig.colorbar(heatmap,fraction=0.03, pad=0.02,
                            ticks=bounds,
                            #extend='both',
                            spacing='uniform',
                            orientation='horizontal',
                            norm=norm,
                            ax=ax)
        cbar.ax.tick_params(rotation=30)
        cbar.ax.set_xlabel(polEmis+'\nRedução méddataShpia da emissão (%)', rotation=0,fontsize=6)
        cbar.ax.get_xaxis().labelpad = 6
        cbar.ax.tick_params(labelsize=7) 
        ax.set_xlim([lonBRAIN.min(), lonBRAIN.max()])
        ax.set_ylim([latBRAIN.min(), latBRAIN.max()])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        shape_path= rootFolder+'/shapefiles/BR_regions.shp'
        #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
        #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
        dataShp = gpd.read_file(shape_path)
        dataShp.boundary.plot(ax=ax,edgecolor='black',linewidth=0.3)
        fig.savefig(os.path.dirname(BASE)+'/figures'+'/ReductionSpatial_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
                   bbox_inches='tight',dpi=300)
        
        
        
        # Por estado
        shape_path= rootFolder+'/shapefiles/Brasil.shp'
        #shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
        #shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
        dataShp = gpd.read_file(shape_path)
        dataBox=[]
        for ii,state in enumerate(dataShp['UF']):
            uf = dataShp[dataShp['UF']==state]
            s,cityMat=dataINshape(lonBRAIN,latBRAIN,uf)
            dataBox.append(q4EMISmatE1[:,0:,cityMat==1][~np.isnan(q4EMISmatE1[:,0:,cityMat==1])])
            try:
                statDf['AveReduction_'+str(pol['Criteria'])][ii] = np.percentile(q4EMISmatE1[:,0:,cityMat==1][~np.isnan(q4EMISmatE1[:,0:,cityMat==1])],50)
                statDf['MaxReduction_'+str(pol['Criteria'])][ii] = np.nanmax(q4EMISmatE1[:,0:,cityMat==1][~np.isnan(q4EMISmatE1[:,0:,cityMat==1])])
            except:
                statDf['AveReduction_'+str(pol['Criteria'])][ii] = 0
                statDf['MaxReduction_'+str(pol['Criteria'])][ii] = 0
                
        statDf.to_csv(os.path.dirname(BASE)+'/tables'+'/statistics_'+pol['tag']+'_'+str(pol['Criteria'])+'.csv')
        fig,ax = plt.subplots()
        bplot1 = ax.boxplot(dataBox,
                   notch=True,  # notch shape
                   vert=True,  # vertical box alignment
                   patch_artist=True)
        ticks = [i+1 for i, v in enumerate(dataShp['UF'])]
        ax.set_xticks(ticks, dataShp['UF'],fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_ylabel(polEmis+'\nRedução da emissão (%)' ,fontsize=8)
        cm = 1/2.54  # centimeters in inches
        fig.set_size_inches(18*cm, 6*cm)
        # fill with colors
        colors = np.repeat(['#71CCF1'],dataShp['UF'].shape[0])
    
        for patch, color in zip(bplot1['boxes'], colors):
            patch.set_facecolor(color)
        fig.savefig(os.path.dirname(BASE)+'/figures'+'/ReductionBox_'+pol['tag']+'_'+str(pol['Criteria'])+'.png', format="png",
                   bbox_inches='tight',dpi=300)
        
        del q4EMISmatE1,q1EMIS,q2EMIS,q3EMIS,q4EMIS,q1BRAIN,q2BRAIN,q3BRAIN,q4BRAIN
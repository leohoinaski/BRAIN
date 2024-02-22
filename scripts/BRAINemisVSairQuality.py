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
#import xarray as xr
import pandas as pd
#import matplotlib.pyplot as plt
import ismember
import geopandas as gpd
import matplotlib
#from scipy.stats import gaussian_kde
#import scipy
#from shapely.geometry import Point
#import pandas as pd
#import matplotlib as mpl
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
  #"criterias" : [260,240,220,200],
  "criterias" : [260],
  "Criteria_average": '1-h average',
}

CO = {
  "Pollutant": "CO",
  "Unit": 'ppb',
  "conv": 1000, # Conversão de ppm para ppb
  "tag":'CO',
  "Criteria_ave": 8,
  "criterias" : [9000],
  "Criteria_average": '8-h average',
}

O3 = {
  "Pollutant": "$O_{3}$",
  "Unit": 'ppm',
  "conv":1962 ,
  "tag":'O3',
  "Criteria_ave": 8,
  #"criterias" : [140,130,120,100],
  "criterias" : [100],
  "Criteria_average": '8-h average',
}

SO2 = {
  "Pollutant": "$SO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 2620,
  "tag":'SO2',
  "Criteria_ave": 24,
  #"criterias" : [125,50,40,30,20],
  "criterias" : [40],
  "Criteria_average": '24-h average',
  
}

PM10 = {
  "Pollutant": "$PM_{10}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM10',
  "Criteria_ave": 24,
  #"criterias" : [120,100,75,50,45],
  "criterias" : [45],
  "Criteria_average": '24-h average',
}

PM25 = {
  "Pollutant": "$PM_{2.5}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM25',
  "Criteria_ave": 24,
  #"criterias" : [60,50,37,25,15],
  "criterias" : [15],
  "Criteria_average": '24-h average',
}

pollutants=[NO2,SO2,O3,PM10,PM25]
pollutants=[NO2]
emisTypes = ['BRAVES','FINN','IND2CMAQ','MEGAN']

#------------------------------PROCESSING--------------------------------------
BASE = os.getcwd()
rootFolder = os.path.dirname(os.path.dirname(BASE))
dataFolder = os.path.dirname(BASE)+'/data'
airQualityFolder =  dataFolder+'/BRAIN'
emissFolder =  dataFolder+'/EMIS'
domain = 'BR'
year = '2020'

shape_pathBR= rootFolder+'/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
dataShpBR = gpd.read_file(shape_pathBR)
br = dataShpBR[dataShpBR['NM_PAIS']=='Brasil']

shape_path= rootFolder+'/shapefiles/Brasil.shp'
#shape_path= '/media/leohoinaski/HDD/shapefiles/SouthAmerica.shp'
#shape_path= '/media/leohoinaski/HDD/shapefiles/BR_Pais_2022/BR_Pais_2022.shp'
dataShp = gpd.read_file(shape_path)
        
print('Looping for each variable')
for kk,pol in enumerate(pollutants):
    criterias = pol['criterias']
    for criteria in criterias:
        pol['Criteria'] = criteria
        # ======== EMIS files============
        # Selecting variable
        if pol['tag']=='CO':
            polEmis = 'CO'
        elif pol['tag']=='O3':
            polEmis = 'NOX'
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
            if len(prefixed)>0:
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
        s,cityMat=BRAINutils.dataINshape(lonBRAIN,latBRAIN,br)
        dataBRAIN[:,:,cityMat==0] = np.nan
        dataEMIS[:,:,cityMat==0] = np.nan
        
        # Removing 1% higher
        dataBRAIN = tst.timeseriesFiltering(dataBRAIN,99)
        #dataEMIS = tst.timeseriesFiltering(dataEMIS,99.9)
        
        # Frequency of violations
        freqExcd= tst.exceedance(dataBRAIN*pol['conv'],pol['Criteria']).astype(float)
        freqExcd[:,cityMat==0] = np.nan
        
        # Figures
        # Average
        legend = pol['Criteria_average'] + ' ' +pol['Pollutant'] +' ('+ pol['Unit'] + ')'
        #cmap = 'YlOrRd'
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["royalblue",'lightskyblue',"azure","yellow","crimson","darkred"])
        BRAINfigs.timeAverageFig(np.nanmax(dataBRAIN.data,axis=0)[0,:,:]*pol['conv'],lonBRAIN,latBRAIN,legend,cmap,
                              dataShp,os.path.dirname(BASE)+'/figures/',pol['tag'],pol['Criteria_average'])
        
        # Exceedence
        legend2 = pol['Criteria_average'] +' ' + pol['Pollutant'] + ' - violations'
        #cmap = 'RdPu'
        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["azure","yellow",'#E72C39',"darkred", 'purple'])
        BRAINfigs.exceedanceFig(freqExcd[0,:,:],lonBRAIN,latBRAIN,legend2,cmap2,
                             dataShp,os.path.dirname(BASE)+'/figures/',pol['tag'],pol['Criteria_average'])
        
        
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
        dataBox=[]
        dataBoxAQ=[]
        dataBoxPixel=[]
        statDf = pd.DataFrame()
        statDf['UF']=dataShp['UF']
        statDf['MAXEMIS'] = np.nan
        statDf['AVEEMIS'] = np.nan
        statDf['NcriticalEvents'] = np.nan
        statDf['NcriticalPixels'] = np.nan
        statDf['AveReduction'] = np.nan
        statDf['MaxReduction'] = np.nan
        
        for ii,state in enumerate(dataShp['UF']):
            uf = dataShp[dataShp['UF']==state]
            sUF,cityMatUF=BRAINutils.dataINshape(lonBRAIN,latBRAIN,uf)
            dataEMISuf = dataEMIS[boolEvents,:,:,:].copy()
            dataBRAINuf = dataBRAIN[boolEvents,:,:,:].copy()
            dataBox.append(dataEMISuf[:,:,cityMatUF==1].flatten()[(dataBRAINuf[:,:,cityMatUF==1].flatten()*pol['conv']>pol['Criteria'])])
            dataBoxAQ.append(dataBRAINuf[:,:,cityMatUF==1].flatten()[(dataBRAINuf[:,:,cityMatUF==1].flatten()*pol['conv']>pol['Criteria'])])
            dataBRAINuf[:,:,cityMatUF==0] = np.nan
            dataBRAINuf[dataBRAINuf*pol['conv']<pol['Criteria']] = np.nan
            dataBoxPixel.append(np.sum(~np.isnan(dataBRAINuf).all(axis=0)))
            try:
                statDf['MAXEMIS'][ii] = np.nanmax(dataEMISuf[:,:,cityMatUF==1].flatten()[(dataBRAINuf[:,:,cityMatUF==1].flatten()*pol['conv']>pol['Criteria'])])
                statDf['AVEEMIS'][ii] = np.percentile(dataEMISuf[:,:,cityMatUF==1].flatten()[(dataBRAINuf[:,:,cityMatUF==1].flatten()*pol['conv']>pol['Criteria'])],50)
                statDf['NcriticalEvents'][ii] = len(dataBRAINuf[:,:,cityMatUF==1].flatten()[(dataBRAINuf[:,:,cityMatUF==1].flatten()*pol['conv']>pol['Criteria'])])
                statDf['NcriticalPixels'][ii] = np.sum(~np.isnan(dataBRAINuf).all(axis=0))
                
            except:
                statDf['MAXEMIS'][ii] = 0
                statDf['NcriticalEvents'][ii] = 0
                statDf['NcriticalPixels'][ii] = 0
                statDf['AVEEMIS'][ii] = 0
            
        del dataBRAINuf, dataEMISuf
        # Figura com estatistica das violações - emissão, número de eventos e número de pixels
        BRAINfigs.exceedingStats(BASE,dataBox,dataShp,pol,polEmis,ds1,dataBoxAQ,dataBoxPixel)
        
        #%%
        #% Encontrando dados em cada quadrante
        #dataBRAINflat = dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']
        #dataEMISflat = dataEMIS[boolEvents,:,:,:].flatten()
        
        # Q1 - BAIXA EMISSÃO E BOA QUALIDADE DO AR
        q1BRAIN = (dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv'])[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']<pol['Criteria']) &  (dataEMIS[boolEvents,:,:,:].flatten()<minMeanEmis)]
        q1EMIS = dataEMIS[boolEvents,:,:,:].flatten()[(dataBRAIN[boolEvents,:,:,:].flatten()*pol['conv']<pol['Criteria']) & (dataEMIS[boolEvents,:,:,:].flatten()<minMeanEmis)]
        
        q1EMISmat = dataBRAIN[boolEvents,:,:,:].copy()
        q1EMISmat[(dataBRAIN[boolEvents,:,:,:]*pol['conv']<pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]<minMeanEmis)]=np.nan
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
        q4BRAINmat2 = dataBRAIN[boolEvents,:,:,:]*pol['conv']
        q4BRAINmat2[~((dataBRAIN[boolEvents,:,:,:]*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]>minMeanEmis))]=np.nan
        
        # EFICIENCIA DE ABATIMENTO ETAPA 1
        # Redução da emissão para os níveis do Q2
        q4EMISmat2 = dataEMIS[boolEvents,:,:,:]
        q4EMISmat2[~((dataBRAIN[boolEvents,:,:,:]*pol['conv']>pol['Criteria']) & (dataEMIS[boolEvents,:,:,:]>minMeanEmis))]=np.nan
        q4EMISmatE1 = ((q4EMISmat2-minMeanEmis)/q4EMISmat2)*100
        
        with open(os.path.dirname(BASE)+'/tables'+'/Q4_'+pol['tag']+'_'+str(pol['Criteria'])+'.npy', 'wb') as f:
            np.save(f, np.sum(freQ4,axis=0).data)
            np.save(f, q4BRAINmat2.data)
            np.save(f, q4EMISmat2.data)
        
        #%%
        
        del ds1, ds ,q1EMISmat,q2EMISmat,q3EMISmat,q4EMISmat,violDf,violAirQ,violEmis
        
        BRAINfigs.QscatterAll(BASE,q1EMIS,q1BRAIN,q2EMIS,q2BRAIN,q3EMIS,q3BRAIN,q4EMIS,q4BRAIN,
                     pol,polEmis,minMeanEmis,dataBRAIN[boolEvents,:,:,:]*pol['conv'],dataEMIS[boolEvents,:,:,:])
        
        del dataBRAIN, dataEMIS,lonBRAINflat,latBRAINflat
        
        # Figura scatter nos quadrantes
        BRAINfigs.Qscatter(BASE,q1EMIS,q1BRAIN,q2EMIS,q2BRAIN,q3EMIS,q3BRAIN,q4EMIS,q4BRAIN,
                     pol,polEmis,minMeanEmis)
        
        # Figura quadrantes no espaço
        BRAINfigs.Qspatial(BASE,rootFolder,lonBRAIN,latBRAIN,freQ1,freQ2,freQ3,freQ4,pol)
        
        # ESTATÍSTICAS Q4 - ETAPA1
        # Por estado
        dataBoxPercentage=[]
        for ii,state in enumerate(dataShp['UF']):
            uf = dataShp[dataShp['UF']==state]
            s,cityMatUF=BRAINutils.dataINshape(lonBRAIN,latBRAIN,uf)
            dataBoxPercentage.append(q4EMISmatE1[:,0:,cityMatUF==1][~np.isnan(q4EMISmatE1[:,0:,cityMatUF==1])])
            try:
                statDf['AveReduction'][ii] = np.percentile(q4EMISmatE1[:,0:,cityMatUF==1][~np.isnan(q4EMISmatE1[:,0:,cityMatUF==1])],50)
                statDf['MaxReduction'][ii] = np.nanmax(q4EMISmatE1[:,0:,cityMatUF==1][~np.isnan(q4EMISmatE1[:,0:,cityMatUF==1])])
            except:
                statDf['AveReduction'][ii] = 0
                statDf['MaxReduction'][ii] = 0
                
        statDf.to_csv(os.path.dirname(BASE)+'/tables'+'/statistics_'+pol['tag']+'_'+str(pol['Criteria'])+'.csv')
        
        # Figura redução no Q4
        BRAINfigs.reductionQ4(BASE,rootFolder,lonBRAIN,latBRAIN,q4EMISmatE1,polEmis,pol,dataBoxPercentage)

        
        del q4EMISmatE1,q1EMIS,q2EMIS,q3EMIS,q4EMIS,q1BRAIN,q2BRAIN,q3BRAIN,q4BRAIN
        
        
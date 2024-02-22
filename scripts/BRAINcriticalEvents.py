#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 08:25:24 2024

@author: leohoinaski
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

NO2 = {
  "Pollutant": "$NO_{2}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1880,
  "tag":'NO2',
  #"Criteria": 260, # 260, 240, 220, 200
  "Criteria_ave": 1,
  #"criterias" : [260,240,220,200],
  "criterias" : [200],
  "Criteria_average": '1-h average',
  'emis':'NOX'
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
  'emis':'NOX'
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
  'emis':'SOX'
  
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
  'emis':'PM10'
}

PM25 = {
  "Pollutant": "$PM_{2.5}$",
  "Unit": '$\u03BCg.m^{-3}$',
  "conv": 1,
  "tag":'PM25',
  "Criteria_ave": 24,
  "criterias" : [60,50,37,25,15],
  "Criteria_average": '24-h average',
  'emis':'PM10'
}

BASE = os.getcwd()
rootFolder = os.path.dirname(os.path.dirname(BASE))
tablesFolder = os.path.dirname(BASE)+'/tables'
domain = 'BR'
year = '2020'

pollutants=[PM25]


for pol in pollutants:
    fileType='statistics_'+pol['tag']
    prefixed = sorted([filename for filename in os.listdir(tablesFolder) if filename.startswith(fileType)])
    dfAll=[]
    for prf in prefixed:
        df = pd.read_csv(tablesFolder+'/'+prf)
        df['standard'] = int(prf.split('_')[2].split('.')[0])
        dfAll.append(df)
    dfAll = pd.concat(dfAll)
    dfAll.sort_values(by="standard", inplace=True,ascending=True)
    #dfAll.set_index(["UF"], inplace=True)
    
    fig,ax = plt.subplots(3,1,sharex=True,gridspec_kw={'wspace':0, 'hspace':0.05})
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches(22*cm, 15*cm)
    cmap = plt.get_cmap('rainbow')
    # Número de eventos críticos
    df_pivot = pd.pivot_table(
    	dfAll,
    	values="NcriticalEvents",
    	index="UF",
    	columns="standard",
    )
    df_pivot=df_pivot.sort_index(axis=1, ascending=False)
    df_pivot.plot(kind="bar",ax=ax[0],cmap=cmap)
    ax[0].set_yscale('symlog')
    ax[0].legend(loc='upper center' ,bbox_to_anchor=(0.5, 1.2),
          fancybox=True, ncol=df_pivot.columns.shape[0],
          fontsize=8, markerscale=15, frameon=False)
    ax[0].set_ylabel('N° eventos críticos\n'+pol['tag'],fontsize=8)
    ax[0].tick_params(axis='both', which='major', labelsize=7)
    # Número de pixels críticos
    df_pivot = pd.pivot_table(
    	dfAll,
    	values="NcriticalPixels",
    	index="UF",
    	columns="standard",
    )
    df_pivot=df_pivot.sort_index(axis=1, ascending=False)
    df_pivot.plot(kind="bar",ax=ax[1],cmap=cmap)
    ax[1].set_yscale('symlog')
    ax[1].set_ylabel('N° pixels críticos\n'+pol['tag'],fontsize=8)
    ax[1].get_legend().remove()
    ax[1].tick_params(axis='both', which='major', labelsize=7)
    # Redução média
    df_pivot = pd.pivot_table(
    	dfAll,
    	values="AveReduction",
    	index="UF",
    	columns="standard",
    )
    df_pivot=df_pivot.sort_index(axis=1, ascending=False)
    df_pivot.plot(kind="bar",ax=ax[2],cmap=cmap)
    #ax[2].set_yscale('symlog')
    ax[2].set_ylim([0,100])
    ax[2].set_ylabel('Redução média\n'+pol['emis'],fontsize=8)
    ax[2].get_legend().remove()
    ax[2].tick_params(axis='both', which='major', labelsize=7)
    fig.savefig(os.path.dirname(BASE)+'/figures'+'/criticalEvents_'+pol['tag']+'.png', format="png",
               bbox_inches='tight',dpi=300)













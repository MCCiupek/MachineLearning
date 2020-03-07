import pandas as pd
import datetime as dt

import sys
sys.version
import numpy as np
import matplotlib.pyplot as plt
import math

filename_actu = 'actu.csv'
filename_cac = 'cac.csv'

CAC40 = pd.read_csv(filename_cac, sep=';', index_col='Unnamed: 0', encoding='latin-1', decimal=',')
data_actu = pd.read_csv(filename_actu, sep=';', index_col='Unnamed: 0', encoding='latin-1', decimal=',')

CAC40.head()

ticker = CAC40.TICKER.unique()

rmdt=[]
for t in ticker:
    actif=CAC40[CAC40.TICKER==t]
    actif=actif.sort_values(['annee','mois','jour'], ascending=[1,1,1])
    rdmt= [math.log(x) for x in actif.CL.iloc[5:].values/actif.CL.iloc[:(actif.shape[0]-5)].values]
    hist_s= [math.log(x) for x in actif.CL.iloc[:(actif.shape[0]-5)].values/actif.CL.iloc[5:].values]
    hist_j= [math.log(x) for x in actif.CL.iloc[:(actif.shape[0]-1)].values/actif.CL.iloc[1:].values]
    actif = actif.iloc[5:]
    actif["RMDT_S"] = rdmt
    actif["HIST_S"] = hist_s
    actif["HIST_J"] = hist_j[4:]
    rmdt.append([t, actif])


aca = rmdt[0][1]
for t, actif in rmdt[1:]:
    actif = actif[['jour', 'mois', 'annee', 'HIST_S', 'HIST_J']]
    actif = actif.rename(columns ={'jour':'jour', 'mois':'mois', 'annee':'annee',"HIST_S":t+"_HIST_S", "HIST_J":t+"_HIST_J"})
    aca = aca.merge(actif, how='left')
    
print(aca.columns[aca.isna().any()].tolist())

# Ajouter des variables
# Variables catégorielles
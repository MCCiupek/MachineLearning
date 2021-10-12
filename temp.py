# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:56:15 2020
@author: paulg
"""
import sys
sys.version
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv
import seaborn as sns

######################## DATA TREATMENT #############################################################
# Recupération des données
filename = "./cac40_v3.csv"
data = pd.read_csv(filename, quotechar='\"', doublequote=False,quoting=csv.QUOTE_NONE).drop(columns=['"'])
data=data.replace('\"','',regex=True)
data.columns = data.columns.str.replace('\"','')
data.head()
all_words=data.columns[24:]

#Affichage des caractéristiques
print("Nombre de lignes : {}\nNombre de colonnes : {}\n".format(len(data), len(data.columns)))
data['recommandation'] = pd.to_numeric(data['recommandation'])
print(data.dtypes)
list_tickers=data.TICKER.unique().tolist()

# Quelques statistiques de la base
stats_df=pd.DataFrame(index=list_tickers)
stats_df["RDM_MOYEN_J"]=[data[data.TICKER == t].RDMT_J.values.mean()*252 for t in list_tickers]
stats_df['MOST_FREQ_WORD']=[data[data.TICKER==t][all_words].sum().argmax() for t in list_tickers]
stats_df['NB_WORD']=[data[data.TICKER==t][all_words].sum().max() for t in list_tickers]
stats_df['RDT_WORD_J']= [(data[data.TICKER==t][stats_df['MOST_FREQ_WORD'][list_tickers.index(t)]]*data[data.TICKER==t].RDMT_J).sum()/stats_df['NB_WORD'][list_tickers.index(t)] for t in list_tickers]
stats_df['RDT_WORD_S']= [(data[data.TICKER==t][stats_df['MOST_FREQ_WORD'][list_tickers.index(t)]]*data[data.TICKER==t].RDMT_S).sum()/stats_df['NB_WORD'][list_tickers.index(t)] for t in list_tickers]
stats_df['RDT_WORD_M']= [(data[data.TICKER==t][stats_df['MOST_FREQ_WORD'][list_tickers.index(t)]]*data[data.TICKER==t].RDMT_M).sum()/stats_df['NB_WORD'][list_tickers.index(t)] for t in list_tickers]

#Récupération des mots qui apparaissent plus de 400 fois
list_words=[]
for w in all_words:
    if data[w].sum()>=400:
        list_words.append(w)

# calcul du rendement  moyen pour chaque mot
result=[]
for w in list_words:
    apparitions=sum(data[w].values)
    rdmt_moy_m=sum(data[w].values*data['RDMT_M'].values)/apparitions
    if rdmt_moy_m >=0.01:
        result.append([w,apparitions,rdmt_moy_m])

# ==============Sortie Tableau énoncé==========================================
# print("Mot\tApparition\tRdt mensuel moyen")
# print("==================================")
# for i in range(len(result)):
#     print("\n{}\t{}\t{}".format(result[i][0],result[i][1],result[i][2].round(4)))
# =============================================================================

#filtre si mots apparus
df = pd.DataFrame(result,columns=['WORD','APPARITIONS','RETURN'])
indic = data.filter(items=df.WORD).sum(axis=1) > 0
data['indic'] = indic
filtered_data = data[data['indic']==True]
words = filtered_data.filter(items=df.WORD)
corr_w = abs(words.corr())

#Affichage graphique de correl
plt.figure(figsize=(12,10))
sns.heatmap(corr_w, annot=False, cmap=plt.cm.Reds)
plt.show()

#Exclusion des variables trop corrélées avec d'autres (si correl > 0.75)
columns = np.full((corr_w.shape[0],), True, dtype=bool)
for i in range(corr_w.shape[0]):
    for j in range(i+1, corr_w.shape[0]):
        if corr_w.iloc[i,j] >= 0.75:
            if columns[j]:
                columns[j] = False
selected_columns = words.columns[columns]
words = words[selected_columns]
filtered_data = filtered_data[filtered_data.columns[1:24]].join(words)
filtered_data['annee']= pd.to_datetime(filtered_data.annee*10000+filtered_data.mois*100+filtered_data.jour,format='%Y%m%d')
filtered_data.rename(columns={'annee': 'date'}, inplace=True)
filtered_data.drop(columns=['mois', 'jour'],axis='columns',  inplace=True)


##################################### REGRESSION MODEL ######################################

import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

list_input=words.columns.tolist()
list_input.append('HISTO_M')
X=filtered_data[list_input]
y=filtered_data.RDMT_M.apply(lambda x: 1 if x >= 0.02 else 0) # variable bianire à expliquer (signal d'achat)

#Splitting the dataset into  training and validation sets
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

# Run the search (X-Validation (cv)= 4)
xgb_clf=xgb.XGBClassifier(booster='gbtree',objective='binary:hinge',n_estimators=1000,eta=0.01)

parameter_space={'eta': [0.01,0.05,0.1],
                 'max_depth': range(1,11,1),
                    'min_child_weight' : [ 1, 3, 5, 7 ],
                    'gamma' : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]}

# on fait l'hyperparamétrage selon le scoring AUC
clf = GridSearchCV(xgb_clf, parameter_space, n_jobs=5, cv=4,scoring='roc_auc')
clf.fit(X_train, y_train)

#calcul des prévisions
y_pred=clf.predict(X_test)
print('Best parameters found:\n', clf.best_params_)
conf=confusion_matrix(y_test,y_pred)
print(conf)
print('AUC Score :{}\n'.format(sklearn.metrics.roc_auc_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))

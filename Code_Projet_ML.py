# -*- coding: utf-8 -*-
"""
Created on Wed May 18 22:56:15 2020

@author: paulg
"""
import sys
sys.version
import numpy as np
import matplotlib.pyplot as plt
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

#Affichage des caractéristiques
print("Nombre de lignes : {}\nNombre de colonnes : {}\n".format(len(data), len(data.columns)))
data['recommandation'] = pd.to_numeric(data['recommandation']) #on convertit en numeric la donnée de l'apparition du mot 'recommandation'
print(data.dtypes)

#On récupère la liste des tickers
list_tickers=data.TICKER.unique().tolist()
#Modification de la base ( date et ajout de données sur les varaitions futures de volume)
data['annee']= pd.to_datetime(data.annee*10000+data.mois*100+data.jour,format='%Y%m%d')
data.rename(columns={'annee': 'date'}, inplace=True)
data.drop(columns=['mois', 'jour'],axis='columns',  inplace=True)
data=data.sort_values(by=['TICKER','date'],ascending=[1,1])
#Calcul de la variation future (à 1j, 1semaine, 1mois) du volume futur pour chaque ticker
CHG_VO_J=[data[data.TICKER == t].VO.shift(-1)/data[data.TICKER == t].VO-1 for t in list_tickers]
CHG_VO_S=[data[data.TICKER == t].VO.shift(-5)/data[data.TICKER == t].VO-1 for t in list_tickers]
CHG_VO_M=[data[data.TICKER == t].VO.shift(-20)/data[data.TICKER == t].VO-1 for t in list_tickers]
#Calcul de la médiane du volume écahangé pour chaque ticker, et du quantile à 75%
MEDIAN_VO=[data[data.TICKER == t].VO*0+data[data.TICKER == t].VO.median() for t in list_tickers]
quantile_VO=[data[data.TICKER == t].VO*0+data[data.TICKER == t].VO.quantile(0.75) for t in list_tickers]
#Ajout de ces données calculées à notre base de data
data['FUTUR_VO_J']=pd.concat(CHG_VO_J)
data['FUTUR_VO_S']=pd.concat(CHG_VO_S)
data['FUTUR_VO_M']=pd.concat(CHG_VO_M)
data['MEDIAN_VO']=pd.concat(MEDIAN_VO)
data['75_CENT_VO']=pd.concat(quantile_VO)

#Pour plus de lisibilité on replace les nouvelles données sur le volume avant celles sur les apparitions des mots
cols= list(data.columns.values)
data=data[cols[0:22]+cols[-5:]+cols[22:-5]]
#On récupère la liste de tous les mots
all_words=data.columns[27:]

# Quelques statistiques de la base (calculées par Ticker)
stats_by_ticker=pd.DataFrame(index=list_tickers)  
#rendement quotidien moyen par ticker
stats_by_ticker["RDM_MOYEN_M"]=[data[data.TICKER == t].RDMT_M.values.mean() for t in list_tickers]    
#mot le plus apparus par ticker
stats_by_ticker['MOST_FREQ_WORD']=[data[data.TICKER==t][all_words].sum().argmax() for t in list_tickers]
#nombre d'apparition du mot le plus souvent apparus par ticker
stats_by_ticker['NB_WORD']=[data[data.TICKER==t][all_words].sum().max() for t in list_tickers]

#Stats classified by words (no more classified by ticker)
stats_by_word=pd.DataFrame(index=all_words)
#nombre d'apparitions du mot
stats_by_word['APPARITIONS']=[data[w].sum() for w in all_words]
#rendement moyen lorsque le mot est cité
stats_by_word["RDM_MOYEN_M"]=[(data[data[w]==1].RDMT_M).mean() for w in all_words]
#frequence hausse rendement mensuel (entre historique et futur) lorsque le mot est cité
stats_by_word['HAUSSE_RDMT_M']=[(data[data[w]==1].RDMT_M>data[data[w]==1].HISTO_M).mean() for w in all_words]
#frequence volume traité du jour supérieur à la médiane (et quantile 75%) lorsque le mot est cité
stats_by_word['VO>MEDIAN']=[(data[data[w]==1].VO>data[data[w]==1].MEDIAN_VO).mean() for w in all_words]
stats_by_word['VO>QUANTIL_75']=[(data[data[w]==1].VO>data[data[w]==1]['75_CENT_VO']).mean() for w in all_words]
#frequence hausse historique (et future) du volume traité à 1jour lorsque le mot est cité 
stats_by_word['VO_HISTO_J>0']=[(data[data[w]==1].VOL_J>0).mean() for w in all_words]
stats_by_word['VO_FUTUR_J>0']=[(data[data[w]==1].FUTUR_VO_J>0).mean() for w in all_words]
stats_by_word=stats_by_word.sort_values(by=['APPARITIONS','RDM_MOYEN_M'],ascending=[0,0])

#Save dataframe of statistique in an excel
#writer = pd.ExcelWriter(r'Statistiques de la base.xlsx', engine='xlsxwriter')
#stats_by_word.to_excel(writer, sheet_name='by Word')
#stats_by_ticker.to_excel(writer, sheet_name='by Ticker')
## Close the Pandas Excel writer and output the Excel file.
#writer.save()

#Récupération des mots qui apparaissent plus de 400 fois
list_words=[]
for w in all_words:
    if data[w].sum()>=400:
        list_words.append(w)
# calcul du rendement  moyen pour chacun de ses mots,
result=[]
for w in list_words:
    apparitions=sum(data[w].values)
    rdmt_moy_m=sum(data[w].values*data['RDMT_M'].values)/apparitions
	# on cnserve ceux pour lesquels le rendement moyen > 1%
    if rdmt_moy_m >=0.01:
        result.append([w,apparitions,rdmt_moy_m])
# Sortie Tableau énoncé
print("Mot\tApparition\tRdt mensuel moyen")
print("==================================")
for i in range(len(result)):
    print("\n{}\t{}\t{}".format(result[i][0],result[i][1],result[i][2].round(4)))
#DataFrame contenant la liste des mots filtrés, leurs rendements moyens et leurs nombres d'apparitions
df = pd.DataFrame(result,columns=['WORD','APPARITIONS','RETURN'])
#On crée une variable indicatrice sur la condition d'apparition d'un des mots de la list de df.WORD
indic = data.filter(items=df.WORD).sum(axis=1) > 0
data['indic'] = indic
#filtered_data contient seulement les lignes pour lesquelles il y a l'apparition d'un mot de la liste df.WORD
filtered_data = data[data['indic']==True]
words = filtered_data.filter(items=df.WORD)
#on calcule la corrélation sur ces lignes filtrées entre les apparitions de chaque mot de df.WORD
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
                
#On exclue les mots trop corrélés avec d'autres de notre liste de mots
selected_columns = words.columns[columns] 
words = words[selected_columns]

#Ici il nous rest les mots plus toutes les données de marché de la base hormis les ticker
filtered_data = filtered_data[filtered_data.columns[1:27]].join(words)

##################################### REGRESSION MODEL ######################################
import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

#nos inputs sont composés à partir des lignes filtrées
# - des mots filtrés après retrait des corrélations élevées 
# - des données de variation du volume traité par rapport à la veille
# - du niveau de volume traité du jour
list_input=words.columns.tolist()
list_input+=['HISTO_M','VO','VOL_J']
X=filtered_data[list_input]

# 2 variables à expliquer pour lesquelles on obtient un AUC correct:	
    #- RDMT supérieur à 2%, et (volume supérieure au quantile 75% ou Volume Futur qui augmente d'au moins 75%)
y2=filtered_data.RDMT_M.apply(lambda x : 1 if x>= 0.02 else 0)*\
 ((filtered_data.FUTUR_VO_J).apply(lambda x : 1 if x>0.75  else 0)+\
  (filtered_data.VO-filtered_data['75_CENT_VO']).apply(lambda x : 1 if x> 0 else 0)-\
  ((filtered_data.FUTUR_VO_J).apply(lambda x : 1 if x>0.75 else 0)*\
  (filtered_data.VO-filtered_data['75_CENT_VO']).apply(lambda x : 1 if x> 0 else 0)))

#    - RDMT supérieur à celui du mois précédent et rendement futur supérieur à 2%
y3=(filtered_data.RDMT_M-filtered_data.HISTO_M).apply(lambda x : 1 if x>= 0 else 0)*(filtered_data.RDMT_M.apply(lambda x : 1 if x>0.02 else 0))

list_y=[y2,y3]

#PLOTING USING plotly
import plotly.express as px
from plotly.offline import plot
filtered_data['axis']=range(1,len(filtered_data)+1)
filtered_data['indic_y2']=y2
filtered_data['indic_y3']=y3
fig_y2= px.bar(filtered_data, x='axis', y='RDMT_M', color='indic_y2',opacity=1)
plot(fig_y2)
fig_y3 = px.bar(filtered_data, x='axis', y='RDMT_M', color='indic_y3',opacity=1)
plot(fig_y3)

# initialisation du modèle de départ utilisé pour les hyperparamétrages
start_model = xgb.XGBClassifier(silent=False, 
								learning_rate=0.2, 
								n_estimators=200, 
								objective='binary:logistic',
								subsample = 1,
								colsample_bytree = 1,
								nthread=4,
								scale_pos_weight=1, random_state=1,
								seed=1)
#Grille d'hyperparam pour la varaible y2
parameter_space_y2={'max_depth':[6,9,10],
                 'min_child_weight':[1,2,3],
			     'gamma':[0,0.4,1],
				 'subsample':[0.9,1],
				 'colsample_bytree':[0.9,1],
				 'reg_alpha':[0,1],			
				 'learning_rate':[0.02], 
				 'n_estimators':[500]}
#Grille d'hyperparam pour la varaible y3
parameter_space_y3={'max_depth':[6,9,10],
                 'min_child_weight':[0,1,1.5],
			     'gamma':[0,1],
				 'subsample':[0.7,0.95,1],
				 'colsample_bytree':[0.7,0.95,1],
				 'reg_alpha':[0,1],			
				 'learning_rate':[0.02], 
				 'n_estimators':[500]}
list_grid=[parameter_space_y2,parameter_space_y3]


# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\ 
do_tuning=False #do_tuning doit être False pour éviter de lancer l'hyperparamétrage, environ (1h10)
# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
#nos 2 modèles issus de l'hyperparamétrage (permet de lancer le code sans refaire l'hyperparamétrage)
tuned_model_y2=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=1,
              learning_rate=0.02, max_delta_step=0, max_depth=6,
              min_child_weight=2, missing=None, n_estimators=500, n_jobs=1,
              nthread=4, objective='binary:logistic', random_state=1,
              reg_alpha=1, reg_lambda=1, scale_pos_weight=1, seed=1,
              silent=False, subsample=1, verbosity=1)

tuned_model_y3=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=1,
              learning_rate=0.02, max_delta_step=0, max_depth=10,
              min_child_weight=0, missing=None, n_estimators=500, n_jobs=1,
              nthread=4, objective='binary:logistic', random_state=1,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=1,
              silent=False, subsample=1, verbosity=1)
tuned_model=[tuned_model_y2,tuned_model_y3]


dic_result=[]
i=0
#Boucle pour l'hyperparamétrage sur les 2 variables à expliquer
for y in list_y:
	print('\t Model for y{} :'.format(i+2))
	print('===========================')
	 #Splitting the dataset into  training and validation sets
	X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=7)
	#normalisation des données
	from sklearn.preprocessing import StandardScaler
	scaler=StandardScaler()
	scaler.fit(X_train)
	X_train=scaler.transform(X_train)
	X_test=scaler.transform(X_test)
	eval_set = [(X_train, y_train), (X_test, y_test)]
	eval_metric = ["error","auc"]
	##################################### MODELE DE DEPART : un premier exemple
	#entraînement du modèle
	start_model.fit(X_train, y_train, eval_metric=eval_metric,eval_set=eval_set, verbose=False)
	#prévision test du modèle
	y_hat=start_model.predict(X_test)
	print('============= START MODEL for (y=y{}) ============\n'.format(i+2))
	print('AUC Score :{}\n'.format(sklearn.metrics.roc_auc_score(y_test,start_model.predict_proba(X_test)[:,1])))
	print(classification_report(y_test,y_hat))
	###########################  HYPERPARAMETRAGE ###########################  
	#Set features and parameters of hyperparameters tuning
	if do_tuning:
		scoring=['roc_auc','recall'] 
		refit='roc_auc' 
		parameter_space=list_grid[i]
		#Set tuning
		clf = GridSearchCV(start_model, param_grid=parameter_space,n_jobs=-1, cv=4,scoring=scoring,verbose=3,return_train_score=True,refit=refit)
		#Launch tuning
		clf.fit(X_train,y_train)
		model=clf.best_estimator_
	else:
		 model=tuned_model[i] #récupération en dur du modèle issu de l'hyperparamétrage
		 model.fit(X_train,y_train)
	y_pred=model.predict(X_test)
	dic_temp={'estimator':model,'name':'y{}'.format(i+2),
		   'X_train':X_train,'y_train':y_train,'X_test':X_test,'y_test':y_test}
	#Display result
	print( '=============== TUNED MODEL (y = y{}) ================='.format(i+2))
	print('Parameters:\n', model.get_params())
	auc=sklearn.metrics.roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
	recall=sklearn.metrics.recall_score(y_test,y_pred)
	precision=sklearn.metrics.precision_score(y_test,y_pred)
	print('\nAUC Score :{}\n'.format(auc))
	print(classification_report(y_test,y_pred))
	add_result={'AUC':auc,'recall':recall,'precision':precision,'y_pred':y_pred}
	dic_temp.update(add_result)
	#adding model and prediction result to  a list of dictionnary
	dic_result.append(dic_temp)
	i+=1

#Comparison of the own model of each variable y
print("y\tAUC\tRecall\tPrecision \n=====================================")
for i in range(len(list_y)):
	print("y{}\t{}\t{}\t{}\n".format(i+2,dic_result[i]['AUC'].round(2),dic_result[i]['recall'].round(2),dic_result[i]['precision'].round(2)))

##### RESULT
# Select the best variable y and its tuned model, here the best seems y3 
k=1 #(0 for y2 and 1 for y3)
y=list_y[k]
best_model=xgb.XGBClassifier(**dic_result[k]['estimator'].get_params())
X_train=dic_result[k]['X_train']
X_test=dic_result[k]['X_test']
y_train=dic_result[k]['y_train']
y_test=dic_result[k]['y_test']
eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["error","auc"]
best_model.set_params(learning_rate=0.02,n_estimators=5000)
best_model.fit(X_train, y_train, eval_metric=eval_metric,eval_set=eval_set,early_stopping_rounds=500,verbose=False)
y_pred=best_model.predict(X_test)
#Display Result
print( '============== BEST MODEL : choice y=y{} ==================\n'.format(k+2))
conf=confusion_matrix(y_test,y_pred)
print('Matrice de confusion:\n',conf)
print('\nAUC Score :{}\n'.format(sklearn.metrics.roc_auc_score(y_test,best_model.predict_proba(X_test)[:,1])))
print(classification_report(y_test,y_pred))
#Plotting AUC in terms of estimator number
results = best_model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.xlabel('Estimator n-th')
plt.title('XGBoost AUC (y{})'.format(k+2))
plt.show()


# VARIABLE IMPORTANCE
features_names=['f{} = {}'.format(i,X.columns.tolist()[i]) for i in range(len(X.columns.tolist()))]
print( '\nFEATURES IMPORTANCE :')
print( '\nFEATURES NAMES :\n',features_names)
#Plot Top 10 importance input variables (features)
fig = plt.figure(figsize=(12,7))
ax_cover = fig.add_subplot(121)
xgb.plot_importance(best_model,max_num_features=10,importance_type='cover',height=0.5, title='Feature Importance (Cover)',ax=ax_cover,show_values=False)
ax_cover.grid(b=None)
ax_gain=fig.add_subplot(122)
xgb.plot_importance(best_model,max_num_features=10,importance_type='gain',height=0.5, title='Feature Importance (Gain)',ax=ax_gain,show_values=False)
ax_gain.grid(b=None)
fig.show()
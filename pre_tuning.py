#============ PRE TUNING SCRIPT=================================
# PRE TUNING : SUCCESSIVE TUNING
# Target : To obtain an idea of an adapted grid for each variable y
scoring=['roc_auc','recall']
refit='recall'
#Tuning max_depth and min_child_weight
parameter_space1={'max_depth':range(11),
                 'min_child_weight':range(6)}
#colsample_bytree= 1, gamma=1, max_delta_step =0, max_depth= 6, min_child_weight= 1, subsample= 0.8)
# on fait l'hyperparamétrage selon le scoring AUC

clf1 = GridSearchCV(start_model, param_grid=parameter_space1,n_jobs=-1, cv=4,scoring=scoring,verbose=3,return_train_score=True,refit=refit)
clf1.fit(X_train, y_train, verbose=True)
print(clf1.best_params_, "\n auc score =",sklearn.metrics.roc_auc_score(y_test,clf1.best_estimator_.predict_proba(X_test)[:,1]))
print('recall score =',sklearn.metrics.recall_score(y_test,clf1.best_estimator_.predict(X_test)))
print('precision score =',sklearn.metrics.precision_score(y_test,clf1.best_estimator_.predict(X_test)))

#here it depends of results obtain from last tuning, with this second tuning on same hyperparam we searhc more accuraccy
parameter_space1_bis={'max_depth':[10,11,12],
					  'min_child_weight':[0,0.5,1,1.5]}
clf1_bis= GridSearchCV(clf1.best_estimator_, param_grid=parameter_space1_bis,n_jobs=-1, cv=4,scoring=scoring,verbose=3,return_train_score=True,refit=refit)
clf1_bis.fit(X_train, y_train, verbose=True)
print(clf1_bis.best_params_, "\n auc score =",sklearn.metrics.roc_auc_score(y_test,clf1_bis.best_estimator_.predict_proba(X_test)[:,1]))
print('recall score =',sklearn.metrics.recall_score(y_test,clf1_bis.best_estimator_.predict(X_test)))
print('precision score =',sklearn.metrics.precision_score(y_test,clf1.best_estimator_.predict(X_test)))

#Tuning gamma
parameter_space2={'gamma':[i/10 for i in range(21)]}
clf2=GridSearchCV(clf1_bis.best_estimator_, param_grid=parameter_space2,n_jobs=-1, cv=4,scoring=scoring,verbose=3,return_train_score=True,refit=refit)
clf2.fit(X_train, y_train, verbose=True)
print(clf2.best_params_, "\n auc score =",sklearn.metrics.roc_auc_score(y_test,clf2.best_estimator_.predict_proba(X_test)[:,1]))
print('recall score =',sklearn.metrics.recall_score(y_test,clf2.best_estimator_.predict(X_test)))
print('precision score =',sklearn.metrics.precision_score(y_test,clf1.best_estimator_.predict(X_test)))

#Tuning subsample and colsample_bytree
parameter_space3 = {'subsample':[i/20.0 for i in range(10,21)],
				    'colsample_bytree':[i/20.0 for i in range(10,21)]}
clf3=GridSearchCV(clf2.best_estimator_, param_grid=parameter_space3,n_jobs=-1, cv=4,scoring=scoring,verbose=3,return_train_score=True,refit=refit)
clf3.fit(X_train, y_train, verbose=True)
print(clf3.best_params_, "\n auc score =",sklearn.metrics.roc_auc_score(y_test,clf3.best_estimator_.predict_proba(X_test)[:,1]))
print('recall score =',sklearn.metrics.recall_score(y_test,clf3.best_estimator_.predict(X_test)))
print('precision score =',sklearn.metrics.precision_score(y_test,clf3.best_estimator_.predict(X_test)))

#Tuning reg_alpha
parameter_space4 = {'reg_alpha':[i/2 for i in range(11)]}
clf4=GridSearchCV(clf3.best_estimator_, param_grid=parameter_space4,n_jobs=-1, cv=4,scoring=scoring,verbose=3,return_train_score=True,refit=refit)
clf4.fit(X_train, y_train, verbose=True)
print(clf4.best_params_, "\n auc score =",sklearn.metrics.roc_auc_score(y_test,clf4.best_estimator_.predict_proba(X_test)[:,1]))
print('recall score =',sklearn.metrics.recall_score(y_test,clf4.best_estimator_.predict(X_test)))
print('precision score =',sklearn.metrics.precision_score(y_test,clf4.best_estimator_.predict(X_test)))

y_pred=clf4.predict(X_test)
print('sum=',y_pred.sum())
print('Best parameters:\n', clf4.best_estimator_)
conf=confusion_matrix(y_test,y_pred)
print(conf)
print('AUC Score :{}\n'.format(sklearn.metrics.roc_auc_score(y_test,clf4.predict_proba(X_test)[:,1])))
print(classification_report(y_test,y_pred))

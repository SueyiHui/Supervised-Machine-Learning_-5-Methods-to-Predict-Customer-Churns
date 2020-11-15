# [Shuyi, Hui]
# [20198085]
# [MMA]
# [2021W]
# [MMA869]
# [07/08/2020]


# Answer to Question [1], Part [1]


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
import pandas_profiling

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import itertools

import scipy

import sklearn


#################################################################### Answer to Question [7], Part [a]###############################################################################################
## Read in data from Uncle Steve's GitHub repository
df = pd.read_csv("https://raw.githubusercontent.com/stepthom/sandbox/master/data/OJ.csv")

##Checking for missing value, and drop them if there is
df.dropna(inplace=True)
##Checking for NaN entry
np.where(pd.isnull(df))
###There is no missing value and NaN Entry.

##Generate Information About the Dataset(EDA)
df.info()
df.head()
pp = pandas_profiling.ProfileReport(df)
pp.to_file('profile_report.html')

###According to the result from PandaProfiling:
##Drop the first column because is it ID counts
##Drop DiscCH and DiscMM due to their high correlation with PctDiscCH and PctDiscMM
##Drop SalePriceCH and SalePriceMM due to their high correlation with PctDiscCH and PctDiscMM
##Drop STORE and Store7 because they are overlapping with StoreID 
##Drop PriceDiff because due to its high correlation with PctDiscMM
df = df[df.columns[1:]]
print(df.columns)

df = df.drop(['DiscCH','DiscMM','SalePriceCH','SalePriceMM','STORE','Store7','PriceDiff'], axis=1)
df.info()
df.head()
###1070 instance, 11 features left:'Purchase', 'WeekofPurchase', 'StoreID', 'PriceCH', 'PriceMM', 'SpecialCH', 'SpecialMM', 'LoyalCH', 'PctDiscMM', 'PctDiscCH','ListPriceDiff'

##Create columns for each stores by splitting StoreID
df = pd.get_dummies(df, columns=['StoreID'], prefix=['StoreID'])
###Strore 5&6 does not exit in the original file >> No sales in store 5&6
pp1 = pandas_profiling.ProfileReport(df)
pp1.to_file('profile_report2.html')

##Data Standatdization
X = df[list(df)[1:]].copy()

scaler = StandardScaler()
features = list(X)[:-5]
X[features] = scaler.fit_transform(X[features])
X.head()

##Change CH to 1 and MM to 0 in the Purchace Column
y = df[['Purchase']].copy()

nums_convert = {"Purchase":{"CH": 1, "MM": 0}}
y.replace(nums_convert, inplace=True)
y.head() 

############################################################################### Answer to Question [7], Part [b] ####################################################################################################
##Splitting the Data Into a 20/80 Split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


## Hepler Function for Plotting ROC
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Adopted from: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

def plot_roc(clf, X_test, y_test, name, ax, show_thresholds=False):
    y_pred_rf = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, y_pred_rf)

    ax.plot([0, 1], [0, 1], 'k--');
    ax.plot(fpr, tpr, label='{}, AUC={:.2f}'.format(name, auc(fpr, tpr)));
    ax.scatter(fpr, tpr);

    if show_thresholds:
        for i, th in enumerate(thr):
            ax.text(x=fpr[i], y=tpr[i], s="{:.2f}".format(th), fontsize=14, 
                     horizontalalignment='left', verticalalignment='top', color='black',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.1));
        
    ax.set_xlabel('False positive rate', fontsize=18);
    ax.set_ylabel('True positive rate', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=18);
    ax.grid(True);
    ax.set_title('ROC Curve', fontsize=18)
############################################################################### Answer to Question [7], Part [c]########################################################
##1.0_Model Selection_Grid Search_Decision Tree(Tune Parameters)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

treeclf = DecisionTreeClassifier(splitter='best', presort=True, class_weight=None, random_state=42)
parameters = {'criterion':('gini', 'entropy'), 'max_depth':[2, 4, 6, 8, 10], 'min_samples_split':[1,2,3,4,5,10], 'min_samples_leaf':[1,2,4,5,10],
             'max_features':[None, 'auto'], 'max_leaf_nodes':[None, 5, 10, 15], 'min_impurity_decrease':[0, 0.1, 0.2],'class_weight':['balanced', None]}

cv_clf = GridSearchCV(treeclf, parameters, scoring='roc_auc', cv=15, return_train_score=True,error_score=0.0, n_jobs=-1)
cv_clf.fit(X, y)
  
cv_clf.best_params_
cv_clf.best_score_
cv_clf.best_estimator_  
###According to the outcout of grid search，the best practice for Decision Tree is using 'entyopy', max_depth = 4 with score of 0.8868221948 

##1.1_Model Development_Apply the Best Practice to Decision Tree
clf = cv_clf.best_estimator_
clf.fit(X_train, y_train)

y_pred_dt = clf.predict(X_val)
clf.feature_importances_

feature_names = X.columns
class_names = [str(x) for x in clf.classes_]

##Plot ROC for Decision Tree Method(with YellowBrick)
##Manual Plotting
plt.style.use('seaborn-poster');
figure = plt.figure(figsize=(10, 6));   
ax = plt.subplot(1, 1, 1);
plot_roc(clf, X_val, y_val, "Decision Tree (test)", ax)
plot_roc(clf, X_train, y_train, "Decision Tree (train)", ax)
plt.legend(loc='lower right', fontsize=18);
plt.tight_layout();

##Model Visualization
from sklearn.tree import plot_tree

plt.figure(figsize=[20,10]);
plot_tree(clf, filled=True, feature_names = feature_names, label='root', fontsize=14)
plt.show();


##2.0_Model Selection_Grid Search_Naive Bayes(Tune Parameters)
from sklearn.naive_bayes import GaussianNB

parameters = {'var_smoothing': [0,1e-9,1e-8]}

gnb = GaussianNB()

NBgridsearch = GridSearchCV(gnb,parameters,cv=15,scoring='roc_auc',return_train_score=True,error_score=0.0,n_jobs=-1)
NBgridsearch.fit(X_train, y_train)

NBgridsearch.best_params_
NBgridsearch.best_score_
NBgridsearch.best_estimator_
###According to the outcout of grid search，the best practice for Naive Bayes is using 'var_smoothing:0' with score of 0.843779851

##2.1_Model Development_Apply the Best Practice to Naive Bayes
gnb = NBgridsearch.best_estimator_
gnb = gnb.fit(X_train, y_train)
gnb

#Model Parameters
gnb.theta_ # Mean of each feature per class
gnb.sigma_ # Variance of each feature per class

##Plot ROC for Navie Bayes Method
##Manual Plotting
plt.style.use('seaborn-poster');
figure = plt.figure(figsize=(10, 6));   
ax = plt.subplot(1, 1, 1);
plot_roc(gnb, X_val, y_val, "Naive Bayes (test)", ax)
plot_roc(gnb, X_train, y_train, "Naive Bayes (train)", ax)
plt.legend(loc='lower right', fontsize=18);
plt.tight_layout();
plt.savefig("ROC_Naive Bayes.png")


##3.0_Model Selection_Grid Search_KNN(Tune Parameters)
from sklearn.neighbors import KNeighborsClassifier

parameters = {'n_neighbors': [5,15,25,50,100],
               'weights':['uniform','distance'],
              'leaf_size':[10,25,50], 
               'p':[1,3,5],
               'metric':['euclidean','manhattan','chebyshev','minkowski','wminkowski','seuclidean','mahalanobis']}

knn_clf = KNeighborsClassifier()

KNNgridsearch = GridSearchCV(knn_clf,parameters,cv=15,scoring='roc_auc',return_train_score=True,error_score=0.0,n_jobs=-1)
KNNgridsearch.fit(X_train, y_train)

KNNgridsearch.best_score_
KNNgridsearch.best_params_
KNNgridsearch.best_estimator_
###According to the outcout of grid search，the best practice for KNN is using 'euclidean' for n_neighbors as 100 with score of 0.892000199

##3.1_Model Development_Apply the Best Practice to KNN
knn_clf = KNNgridsearch.best_estimator_
knn_clf = knn_clf.fit(X_train,y_train)

##Plot ROC for KNN Method
##Manual Plotting
plt.style.use('seaborn-poster');
figure = plt.figure(figsize=(10, 6));   
ax = plt.subplot(1, 1, 1);
plot_roc(knn_clf, X_val, y_val, "KNN (test)", ax)
plot_roc(knn_clf, X_train, y_train, "KNN (train)", ax)
plt.legend(loc='lower right', fontsize=18);
plt.tight_layout();
plt.savefig("Q7_ROC_KNN.png")


##4.0_Model Selection_Grid Search_XGboost(Tune Parameters_Additional Tryout1)
from xgboost import XGBClassifier

parameters = {'n_estimators': [100,1000],
               'max_depth':[1,3],
              'learning_rate':[0.01,0.1], 
               'gamma':[0,1,5],}

xg_clf = XGBClassifier()

XGgridsearch = GridSearchCV(xg_clf,parameters,cv=15,scoring='roc_auc',return_train_score=True,error_score=0.0,n_jobs=-1)
XGgridsearch.fit(X_train, y_train)

XGgridsearch.best_score_
XGgridsearch.best_params_
XGgridsearch.best_estimator_
###According to the outcout of grid search，the best practice for XGboost is using 'gamma = 0, learning_rate = 0.1' for n_estimators as 100 with score of 0.903891675

##4.1_Model Development_Apply the Best Practice to KNN
xg_clf = XGgridsearch.best_estimator_
xg_clf = xg_clf.fit(X_train,y_train)

##Plot ROC for XGboost Method
##Manual Plotting
plt.style.use('seaborn-poster');
figure = plt.figure(figsize=(10, 6));   
ax = plt.subplot(1, 1, 1);
plot_roc(xg_clf, X_val, y_val, "XGboost (test)", ax)
plot_roc(xg_clf, X_train, y_train, "XGboost (train)", ax)
plt.legend(loc='lower right', fontsize=18);
plt.tight_layout();
plt.savefig("Q7_ROC_XGboost.png")


##5.0_Model Selection_Grid Search_Logistic Regression(Tune Parameters_Additional Tryout2)
from sklearn.linear_model import LogisticRegression

parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
              'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter':[1,50,100,1000],
               'tol':[1e0,1e-1,1e-2,1e-3,1e-4,1e-5],
              'C':[0.0,0.5,1.0,1.5],
              'class_weight':[None,'balanced'],
              'random_state':[42]}

reg_clf = LogisticRegression()

REGgridsearch = GridSearchCV(reg_clf,parameters,cv=15,scoring='roc_auc',return_train_score=True,error_score=0.0,n_jobs=-1)
REGgridsearch.fit(X_train, y_train)

REGgridsearch.best_score_
REGgridsearch.best_params_
REGgridsearch.best_estimator_
###According to the outcout of grid search，the best practice for Logistic Regression is using 'gamma = 0, learning_rate = 0.1' for n_estimators as 100 with score of 0.903891675

##5.1_Model Development_Apply the Best Practice to KNN
reg_clf = REGgridsearch.best_estimator_
reg_clf = reg_clf.fit(X_train,y_train)

##Plot ROC for Logistic Regression Method
##Manual Plotting
plt.style.use('seaborn-poster');
figure = plt.figure(figsize=(10, 6));   
ax = plt.subplot(1, 1, 1);
plot_roc(reg_clf, X_val, y_val, "Logistic Regression (test)", ax)
plot_roc(reg_clf, X_train, y_train, "Logistic Regression (train)", ax)
plt.legend(loc='lower right', fontsize=18);
plt.tight_layout();
plt.savefig("Q7_ROC_Logistic Regression.png")


###Model Performance Comparsion_ Comparing Multiple Algorithms(Performance Summary)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB

print("\n\nDecisionTree")

clf = cv_clf.best_estimator_
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))


print("NB")

gnb = NBgridsearch.best_estimator_
gnb = gnb.fit(X_train, y_train)
y_pred_nb = gnb.predict(X_val)
print(accuracy_score(y_val, y_pred_nb))
print(confusion_matrix(y_val, y_pred_nb))
print(classification_report(y_val, y_pred_nb))

gnb.theta_


print("\n\nKNN")
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNNgridsearch.best_estimator_
knn_clf = knn_clf.fit(X_train,y_train)
y_pred_knn = knn_clf.predict(X_val)
print(accuracy_score(y_val, y_pred_knn))
print(confusion_matrix(y_val, y_pred_knn))
print(classification_report(y_val, y_pred_knn))


print("\n\nXGBoost")
xg_clf = XGgridsearch.best_estimator_
xg_clf = xg_clf.fit(X_train,y_train)
y_pred = xg_clf.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))


print("\n\nLogisticRegression")
reg_clf = REGgridsearch.best_estimator_
reg_clf = reg_clf.fit(X_train,y_train)
y_pred = reg_clf.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
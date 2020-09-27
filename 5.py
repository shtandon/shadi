#oversampl + featr selection

 
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
#import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle
df=pd.read_csv('Dataset.txt', sep='\t')

df.set_index('Index', inplace = True)
#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.isnull().values.any())

#LABELS = ["Not buy", "Buy"]
count_classes = pd.value_counts(df['C'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Buy or Not Buy")
#plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")

Yes = df[df['C']==1]
No = df[df['C']==0]
#print(Yes.shape)
#print(No.shape)

df['F15'] = pd.DatetimeIndex(df['F15']).year

df['F16'] = pd.DatetimeIndex(df['F16']).year


#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
print(top_corr_features)
plt.figure(figsize=(20,20))
#plot heat map
#=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.heatmap(corrmat,annot=True)


X= df.drop(["C"], axis=1)
y= df.C



os =  RandomOverSampler()
X_resOs,y_resOs =os.fit_sample(X,y)

#print('Original dataset shape {}'.format(Counter(y)))
#print('Resampled dataset shape {}'.format(Counter(y_resNm)))



# configure to select all features
fsOs = SelectKBest(score_func=f_classif, k=22)
# learn relationship from training data
fsOs.fit(X_resOs,y_resOs)
# transform train input data
X_train_fsOS = fsOs.transform(X_resOs)
# transform test input data
#X_test_fs = fs.transform(X_test)
#X_train_fs.shape

#print("Score: ",fs.scores_)
#print("pvalues ",fs.pvalues_)

dfscores = pd.DataFrame(fsOs.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
#print(featureScores.nlargest(10,'Score'))

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Specs','Score']
#print(featureScores.nlargest(10,'Score'))


X_fsOS=X_resOs.drop(["F13","F9","F10","F6","F15","F16","F8","F3","F1","F7","F11","F5"], axis =1)
#print(X_fsNm.head())

print("Over+selection-----------")
'''
parameters = {'penalty':['l1', 'l2', 'elasticnet', 'none'],
              'C':range(1,30),
              'fit_intercept':[True,False],
             
              
             }
logreg = LogisticRegression()
cvLogisticRegressionNs =cross_val_score(logreg, X_fsOS,y_resOs, cv=10, scoring ='accuracy').mean()
print("Logistic Regression NS: ",cvLogisticRegressionNs)

grid_searchlog = GridSearchCV(estimator = logreg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search2 = grid_searchlog.fit(X_fsOS, y_resOs)
accuracy = grid_search2.best_score_
print("accuracy:",accuracy)
print("best_params_",grid_search2.best_params_)

'''
knnclassifier = KNeighborsClassifier()
cvKnnclassifier=cross_val_score(knnclassifier, X_fsOS,y_resOs, cv=10, scoring ='accuracy').mean()
print("KNeighborsClassifier: ",cvKnnclassifier)
'''
parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
              'weights':['uniform', 'distance'],
              'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
             
             }
grid_search = GridSearchCV(estimator = knnclassifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 2,
                           n_jobs = -1,
                           )
grid_search1 = grid_search.fit(X_fsOS, y_resOs)
accuracy = grid_search1.best_score_
print("accuracy:",accuracy)
print("best_params_",grid_search1.best_params_)
'''
'''
linearRegression = LinearRegression()
cvLinearRegression=cross_val_score(linearRegression, X_fsNm, y_resNm, cv=10, scoring ='accuracy').mean()
print("Klinear Regressionr: ",cvLinearRegression)


svr = SVR()
cvSVR=cross_val_score(svr, X_fsNm, y_resNm, cv=10, scoring ='accuracy').mean()
print("SVR: ",cvSVR)
'''
'''
oneClassSVM=OneClassSVM()
cvoneClassSVM=cross_val_score(oneClassSVM, X_fsNm, y_resNm, cv=10, scoring ='accuracy').mean()
print("OneClassSVM: ",cvoneClassSVM)

svc = SVC()
cvSVC=cross_val_score(svc, X_fsNm, y_resNm, cv=10, scoring ='accuracy').mean()
print("SVR: ",cvSVC)
'''
'''
decisionTreeClassifier = DecisionTreeClassifier()
cvDecisionTreeClassifier=cross_val_score(decisionTreeClassifier, X_resOs,y_resOs, cv=10, scoring ='accuracy').mean()
print("decisionTreeClassifier: ",cvDecisionTreeClassifier)


randomForestClassifier = RandomForestClassifier()
cvRandomForestClassifier=cross_val_score(randomForestClassifier, X_resOs,y_resOs, cv=10, scoring ='accuracy').mean()
print("RandomForestClassifier: ",cvRandomForestClassifier)

'''

'''
gaussianNB = GaussianNB()
cvGaussianNB=cross_val_score(gaussianNB, X_fsOS,y_resOs, cv=10, scoring ='accuracy').mean()
print("GaussianNB: ",cvGaussianNB)


bernoulliNB = BernoulliNB()
cvBernoulliNB=cross_val_score(bernoulliNB, X_fsOS,y_resOs, cv=10, scoring ='accuracy').mean()
print("cvBernoulliNB: ",cvBernoulliNB)
'''
pickle.dump(cvKnnclassifier,open('data.pkl','wb'))
data_model = pickle.load(open('data.pkl','rb'))




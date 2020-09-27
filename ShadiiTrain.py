#no feature enginerring
#no feature Selection
#no balancing data

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








logreg = LogisticRegression()
cvLogisticRegression =cross_val_score(logreg, X,y, cv=10, scoring ='accuracy').mean()
print("Logistic Regression : ",cvLogisticRegression)



knnclassifier = KNeighborsClassifier(n_neighbors=4)
cvKnnclassifier=cross_val_score(knnclassifier, X,y, cv=10, scoring ='accuracy').mean()
print("KNeighborsClassifier: ",cvKnnclassifier)





'''

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
cvRandomForestClassifier=cross_val_score(randomForestClassifier, X,y, cv=10, scoring ='accuracy').mean()
print("RandomForestClassifier: ",cvRandomForestClassifier)

'''

gaussianNB = GaussianNB()
cvGaussianNB=cross_val_score(gaussianNB, X,y, cv=10, scoring ='accuracy').mean()
print("GaussianNB: ",cvGaussianNB)


bernoulliNB = BernoulliNB()
cvBernoulliNB=cross_val_score(bernoulliNB, X,y, cv=10, scoring ='accuracy').mean()
print("cvBernoulliNB: ",cvBernoulliNB)


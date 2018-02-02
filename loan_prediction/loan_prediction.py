# loan Prediction
"""
Created on Tue Jan 23 10:01:57 2018

@author: A
"""
#Load libraries
import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Shape of the data
print(train.shape)
print(test.shape)

#class distribution
print(train.groupby('Loan_Status').size())

#Dropping Loan_ID
drop_columns = ['Loan_ID']
train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)

# Taking care of missing Gender, Married, Self_Employed, Credit_History,LoanAmount,Loan_Amount_Term, Dependents
train["Gender"].fillna("Male", inplace=True)
test["Gender"].fillna("Male", inplace=True)
train["Married"].fillna("Yes", inplace=True)
train["Self_Employed"].fillna("No", inplace=True)
test["Self_Employed"].fillna("No", inplace=True)
train["Credit_History"].fillna(1, inplace=True)
test["Credit_History"].fillna(1, inplace=True)
train["LoanAmount"].fillna(train["LoanAmount"].median(skipna=True), inplace=True)
test["LoanAmount"].fillna(test["LoanAmount"].median(skipna=True), inplace=True)
train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].median(skipna=True), inplace=True)
test["Loan_Amount_Term"].fillna(test["Loan_Amount_Term"].median(skipna=True), inplace=True)
train["Dependents"].fillna(train["Dependents"].mode()[0], inplace=True)
test["Dependents"].fillna(test["Dependents"].mode()[0], inplace=True)

#histograms
train.hist()
plt.show()

#scatter plot matrix
scatter_matrix(train)
plt.show()

#Split out validation and train sets
X = train.iloc[:, :-1].values
Y = train.iloc[:, -1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 10] = labelencoder_X.fit_transform(X[:, 10])

onehotencoder = OneHotEncoder(categorical_features = [2,10])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size = validation_size, random_state = seed)

#Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train,Y_train)
X_validation = lda.transform(X_validation)

#Test options and evaluation metrics
seed = 7
scoring = 'accuracy'

#Spot check algorithms
models = []
models.append(('LR', LogisticRegression())) 
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
models.append(('DT', DecisionTreeClassifier(criterion = "entropy")))
models.append(('RF', RandomForestClassifier(n_estimators=10, criterion='entropy')))
models.append(('NB', GaussianNB())) 
models.append(('KSVM', SVC(kernel = 'rbf')))
print(models)

#evaluate each model in turn
results = []
names = []
for name, model in models:
    cv_results = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    


# Building the optimal model using Backward Elimination
"""
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((614, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,9,10,11,12,13,14,15,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,9,10,12,13,14,15,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,9,10,13,14,15,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,9,10,13,14,15,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 1, 2, 3, 5,6,9,10,13,14,15,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 1, 2, 5,6,9,10,13,14,15,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 2, 5,6,9,10,13,14,15,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 2, 5,6,9,10,13,14,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary() 

X_opt = X[:, [0, 2,6,9,10,13,14,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary()

X_opt = X[:, [0, 2,6,9,10,13,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary()

X_opt = X[:, [0, 2,6,9,13,16]]
lda_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lda_OLS.summary()
"""
 
#Compare algorithms
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Make predictions on validation set
"""lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
Y_pred = lda.predict(X_validation)
print(accuracy_score(Y_validation, Y_pred))
print(confusion_matrix(Y_validation, Y_pred))
print(classification_report(Y_validation, Y_pred))"""

#Without LDA
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_validation)
print(accuracy_score(Y_validation, Y_pred))
print(confusion_matrix(Y_validation, Y_pred))
print(classification_report(Y_validation, Y_pred))

#With LDA
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_validation)
print(accuracy_score(Y_validation, Y_pred))
print(confusion_matrix(Y_validation, Y_pred))
print(classification_report(Y_validation, Y_pred))

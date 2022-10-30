#!/usr/bin/env python
# coding: utf-8


## Creating model for Titanic problem
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# # Data analysis and cleansing
original_train = pd.read_csv("./train.csv")
original_train.info()
original_train.head()
train = pd.read_csv("./train.csv")
train.head()
train.info()
train.isna().sum()
train["Embarked"].hist(bins = 30, figsize = (10,5))
sns.countplot(x = 'Survived', hue = 'Embarked', data = train)
plt.figure(figsize = (15,10))
plt.grid()
sns.boxplot(y = 'Fare', x = 'Embarked', data = train)
train.groupby('Pclass')['Age'].mean()

# Setting attribute Age
def Fill_Age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isna(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age




train["Age"] = train[["Age","Pclass"]].apply(Fill_Age, axis = 1)
train.drop('Cabin', axis = 1, inplace = True)

# Setting attribute Embarked
def Fill_Embarked(cols):
    Embarked = cols[0]
    Fare = cols[1]
    difference = [abs(Fare - 60), abs(Fare - 27), abs(Fare - 13)]
    min_value = min(difference)
    if pd.isna(Embarked):
        if difference.index(min_value) == 0:
            return 'C'
        elif difference.index(min_value) == 1:
            return 'S'
        else:
            return 'Q'
    else:
        return Embarked            




np.where(train["Embarked"].isnull())[0]
print(train.loc[[829]])
print(train.loc[[61]])
train["Embarked"] = train[["Embarked","Fare"]].apply(Fill_Embarked, axis = 1)
sex = pd.get_dummies(train["Sex"], drop_first = True)
embarked = pd.get_dummies(train["Embarked"], drop_first = True)

train = pd.concat([train,sex,embarked], axis = 1)
train.drop(["PassengerId","Name","Ticket","Sex","Embarked"], axis = 1, inplace = True)
train.head()


# # Train the model

from sklearn.model_selection import train_test_split
X = train.drop('Survived', axis = 1)
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)




from sklearn.linear_model import LogisticRegression
logModel = LogisticRegression(solver = 'newton-cg')
logModel.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000)
rfc_fit = rfc.fit(X_train,y_train)

predicciones = logModel.predict(X_test)
rfc_predicciones = rfc.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, roc_curve
print(classification_report(y_test, predicciones))

print(classification_report(y_test, rfc_predicciones))




confusion_matrix(y_test, predicciones)

confusion_matrix(y_test, rfc_predicciones)




y_pred_prob = logModel.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize = (10,7))
plt.plot(fpr,tpr,color = 'Red', lw = 2, label = 'Curva ROC')
plt.plot([0,1],[0,1], color = 'Blue', lw = 2, linestyle = '--')
plt.xlabel('FPR')
plt.xlabel('TPR')
plt.title('ROC curve')
plt.show()




y_pred_prob = rfc.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize = (10,7))
plt.plot(fpr,tpr,color = 'Red', lw = 2, label = 'Curva ROC')
plt.plot([0,1],[0,1], color = 'Blue', lw = 2, linestyle = '--')
plt.xlabel('FPR')
plt.xlabel('TPR')
plt.title('ROC curve')
plt.show()

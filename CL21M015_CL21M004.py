#Best score code 0.49144
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

df1 = pd.read_csv("Dataset_1_Training.csv",header=None, dtype='unicode')
df2 = pd.read_csv("Dataset_1_Testing.csv",header=None,  dtype='unicode')
df3 = pd.read_csv("Dataset_2_Training.csv",header=None, dtype='unicode')
df4 = pd.read_csv("Dataset_2_Testing.csv",header=None, dtype='unicode')

X1=df1.iloc[1:-2,1:]
X1=X1.T
X1 = StandardScaler().fit_transform(X1)
#print(X1)

X2=df3.iloc[1:-4,1:]
X2=X2.T
X2 = StandardScaler().fit_transform(X2)
#print(X2)

yCo1 = df1.iloc[-2,1:]
yCo1=yCo1.astype('int')
yCo2 = df1.iloc[-1,1:]
yCo2=yCo2.astype('int')
#print(yCo2)


yCo3 = df3.iloc[-4,1:]
yCo3=yCo3.astype('int')
yCo4 = df3.iloc[-3,1:]
yCo4=yCo4.astype('int')
yCo5 = df3.iloc[-2,1:]
yCo5=yCo5.astype('int')
yCo6 = df3.iloc[-1,1:]
yCo6=yCo6.astype('int')
#print(yCo2)

X1_test= df2.iloc[1:,1:]
X1_test= X1_test.T
X1_test = StandardScaler().fit_transform(X1_test)
#print(X1_test)

X2_test= df4.iloc[1:,1:]
X2_test= X2_test.T
#print(X2_test)
X2_test = StandardScaler().fit_transform(X2_test)
#print(X2_test)


model1 = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=800)
model1.fit(X1, yCo1)
predictionsCo1 = model1.predict(X1_test)

model2 = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=1200)
model2.fit(X1, yCo2)
predictionsCo2 = model2.predict(X1_test)

model3 = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=400)
model3.fit(X2, yCo3)
predictionsCo3 = model3.predict(X2_test)

model4 =BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=1600)
model4.fit(X2, yCo4)
predictionsCo4 = model4.predict(X2_test)

model5 = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=800)
model5.fit(X2, yCo5)
predictionsCo5 = model5.predict(X2_test)

model6 = AdaBoostClassifier(base_estimator=svm.SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME')
model6.fit(X2, yCo6)
predictionsCo6 = model6.predict(X2_test)

#print(predictionsCo1)
#print(predictionsCo2)
predictions = np.concatenate((predictionsCo1, predictionsCo2,predictionsCo3,predictionsCo4,predictionsCo5,predictionsCo6))
#print(predictions)
dCo1= pd.DataFrame(predictions)
dCo1.to_csv('CL21M015_CL21M004.csv',header=["Predicted"],index_label='Id')

#The following code is commented which has metrics calculation and might take time to run, hence commented them but they serve as a reference for our work.
'''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.metrics import precision_recall_curve

import pandas as pd

df1 = pd.read_csv("Dataset_1_Training.csv",header=None, dtype='unicode')
df2 = pd.read_csv("Dataset_1_Testing.csv",header=None,  dtype='unicode')
df3 = pd.read_csv("Dataset_2_Training.csv",header=None, dtype='unicode')
df4 = pd.read_csv("Dataset_2_Testing.csv",header=None, dtype='unicode')



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



# k=5
# K-Fold crossvalidation test on the models for all given dataset and descriptor combination and the output results are given as table in written document

from sklearn.model_selection import cross_val_score

np.average(cross_val_score(BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100),X1,yCo1))

np.average(cross_val_score(BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100),X1,yCo2))

np.average(cross_val_score(BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100),X2,yCo3))

np.average(cross_val_score(BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100),X2,yCo4))

np.average(cross_val_score(BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100),X2,yCo5))

np.average(cross_val_score(BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100),X2,yCo6))



np.average(cross_val_score(AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME'),X1,yCo1))

np.average(cross_val_score(AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME'),X1,yCo2))

np.average(cross_val_score(AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME'),X2,yCo3))

np.average(cross_val_score(AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME'),X2,yCo4))

np.average(cross_val_score(AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME'),X2,yCo5))

np.average(cross_val_score(AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME'),X2,yCo6))





np.average(cross_val_score(LogisticRegression(),X1,yCo1))

np.average(cross_val_score(SVC(),X1,yCo1))

np.average(cross_val_score(RandomForestClassifier(n_estimators=40),X1,yCo1))

np.average(cross_val_score(RandomForestClassifier(n_estimators=60),X1,yCo1))

np.average(cross_val_score(RandomForestClassifier(n_estimators=10),X1,yCo1))

np.average(cross_val_score(RandomForestClassifier(n_estimators=10),X1,yCo1))

np.average(cross_val_score(GradientBoostingClassifier(),X1,yCo1))

np.average(cross_val_score(GaussianNB(),X1,yCo1))

np.average(cross_val_score(GaussianNB(),X1,yCo2))

np.average(cross_val_score(LogisticRegression(),X1,yCo2))

np.average(cross_val_score(SVC(),X1,yCo2))

np.average(cross_val_score(RandomForestClassifier(n_estimators=40),X1,yCo2))

np.average(cross_val_score(RandomForestClassifier(n_estimators=10),X1,yCo2))

np.average(cross_val_score(RandomForestClassifier(n_estimators=60),X1,yCo2))

np.average(cross_val_score(GradientBoostingClassifier(),X1,yCo2))





np.average(cross_val_score(GaussianNB(),X2,yCo3))

np.average(cross_val_score(LogisticRegression(),X2,yCo3))

np.average(cross_val_score(SVC(),X2,yCo3))

np.average(cross_val_score(RandomForestClassifier(n_estimators=40),X2,yCo3))

np.average(cross_val_score(RandomForestClassifier(n_estimators=10),X2,yCo3))

np.average(cross_val_score(RandomForestClassifier(n_estimators=60),X2,yCo3))

np.average(cross_val_score(RandomForestClassifier(n_estimators=100),X2,yCo3))

np.average(cross_val_score(GradientBoostingClassifier(),X2,yCo3))

np.average(cross_val_score(GaussianNB(),X2,yCo4))

np.average(cross_val_score(GaussianNB(),X2,yCo5))

np.average(cross_val_score(GaussianNB(),X2,yCo6))

np.average(cross_val_score(LogisticRegression(),X2,yCo4))

np.average(cross_val_score(LogisticRegression(),X2,yCo5))

np.average(cross_val_score(LogisticRegression(),X2,yCo6))

np.average(cross_val_score(SVC(kernel='linear',gamma='auto',C=2),X2,yCo4))

np.average(cross_val_score(SVC(),X2,yCo5))

np.average(cross_val_score(SVC(),X2,yCo6))

np.average(cross_val_score(RandomForestClassifier(n_estimators=40),X2,yCo4))

np.average(cross_val_score(RandomForestClassifier(n_estimators=10),X2,yCo4))

np.average(cross_val_score(RandomForestClassifier(n_estimators=60),X2,yCo4))

np.average(cross_val_score(RandomForestClassifier(n_estimators=100),X2,yCo4))

np.average(cross_val_score(RandomForestClassifier(n_estimators=70),X2,yCo4))

np.average(cross_val_score(RandomForestClassifier(n_estimators=40),X2,yCo5))

np.average(cross_val_score(RandomForestClassifier(n_estimators=10),X2,yCo5))

np.average(cross_val_score(RandomForestClassifier(n_estimators=60),X2,yCo5))

np.average(cross_val_score(RandomForestClassifier(n_estimators=40),X2,yCo6))

np.average(cross_val_score(RandomForestClassifier(n_estimators=10),X2,yCo6))

np.average(cross_val_score(RandomForestClassifier(n_estimators=60),X2,yCo6))

np.average(cross_val_score(GradientBoostingClassifier(),X2,yCo4))

np.average(cross_val_score(GradientBoostingClassifier(),X2,yCo5))

np.average(cross_val_score(GradientBoostingClassifier(),X2,yCo6))








from sklearn.model_selection import train_test_split

x11_train,x11_test,y11_train,y11_test=train_test_split(X1,yCo1,test_size=0.3,random_state=10)  #values will no shuffle for random_state 10
x12_train,x12_test,y12_train,y12_test=train_test_split(X1,yCo2,test_size=0.3,random_state=10)
x23_train,x23_test,y23_train,y23_test=train_test_split(X2,yCo3,test_size=0.3,random_state=10)
x24_train,x24_test,y24_train,y24_test=train_test_split(X2,yCo4,test_size=0.3,random_state=10)
x25_train,x25_test,y25_train,y25_test=train_test_split(X2,yCo5,test_size=0.3,random_state=10)
x26_train,x26_test,y26_train,y26_test=train_test_split(X2,yCo6,test_size=0.3,random_state=10)


#for accuracy score for all the used models using class-balanced datasets and the results are noted in the written report in tabular fashion
classifier = GaussianNB()
classifier.fit(x11_train,y11_train)
y11_predict=classifier.predict(x11_test)
print("accuracy score",accuracy_score(y11_test,y11_predict))

classifier = RandomForestClassifier(n_estimators=60)
classifier.fit(x11_train,y11_train)
y11_predict=classifier.predict(x11_test)
print("accuracy score",accuracy_score(y11_test,y11_predict))

classifier = SVC(kernel='rbf',gamma='auto',C=2)
classifier.fit(x11_train,y11_train)
y11_predict=classifier.predict(x11_test)
print("accuracy score",accuracy_score(y11_test,y11_predict))

classifier = GradientBoostingClassifier()
classifier.fit(x11_train,y11_train)
y11_predict=classifier.predict(x11_test)
print("accuracy score",accuracy_score(y11_test,y11_predict))

classifier = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME')
classifier.fit(x11_train,y11_train)
y11_predict=classifier.predict(x11_test)
print("accuracy score",accuracy_score(y11_test,y11_predict))

classifier = LogisticRegression()
classifier.fit(x11_train,y11_train)
y11_predict=classifier.predict(x11_test)
print("accuracy score",accuracy_score(y11_test,y11_predict))

classifier = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100)
classifier.fit(x11_train,y11_train)
y11_predict=classifier.predict(x11_test)
print("accuracy score",accuracy_score(y11_test,y11_predict))







classifier = GaussianNB()
classifier.fit(x12_train,y12_train)
y12_predict=classifier.predict(x12_test)
print("accuracy score",accuracy_score(y12_test,y12_predict))

classifier = RandomForestClassifier(n_estimators=60)
classifier.fit(x12_train,y12_train)
y12_predict=classifier.predict(x12_test)
print("accuracy score",accuracy_score(y12_test,y12_predict))

classifier = SVC(kernel='rbf',gamma='auto',C=2)
classifier.fit(x12_train,y12_train)
y12_predict=classifier.predict(x12_test)
print("accuracy score",accuracy_score(y12_test,y12_predict))

classifier = GradientBoostingClassifier()
classifier.fit(x12_train,y12_train)
y12_predict=classifier.predict(x12_test)
print("accuracy score",accuracy_score(y12_test,y12_predict))

classifier = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME')
classifier.fit(x12_train,y12_train)
y12_predict=classifier.predict(x12_test)
print("accuracy score",accuracy_score(y12_test,y12_predict))

classifier = LogisticRegression()
classifier.fit(x12_train,y12_train)
y12_predict=classifier.predict(x12_test)
print("accuracy score",accuracy_score(y12_test,y12_predict))

classifier = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100)
classifier.fit(x12_train,y12_train)
y12_predict=classifier.predict(x12_test)
print("accuracy score",accuracy_score(y12_test,y12_predict))





classifier = GaussianNB()
classifier.fit(x23_train,y23_train)
y23_predict=classifier.predict(x23_test)
print("accuracy score",accuracy_score(y23_test,y23_predict))

classifier = RandomForestClassifier(n_estimators=60)
classifier.fit(x23_train,y23_train)
y23_predict=classifier.predict(x23_test)
print("accuracy score",accuracy_score(y23_test,y23_predict))

classifier = SVC(kernel='rbf',gamma='auto',C=2)
classifier.fit(x23_train,y23_train)
y23_predict=classifier.predict(x23_test)
print("accuracy score",accuracy_score(y23_test,y23_predict))

classifier = GradientBoostingClassifier()
classifier.fit(x23_train,y23_train)
y23_predict=classifier.predict(x23_test)
print("accuracy score",accuracy_score(y23_test,y23_predict))

classifier = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME')
classifier.fit(x23_train,y23_train)
y23_predict=classifier.predict(x23_test)
print("accuracy score",accuracy_score(y23_test,y23_predict))

classifier =  LogisticRegression()
classifier.fit(x23_train,y23_train)
y23_predict=classifier.predict(x23_test)
print("accuracy score",accuracy_score(y23_test,y23_predict))

classifier =  BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100)
classifier.fit(x23_train,y23_train)
y23_predict=classifier.predict(x23_test)
print("accuracy score",accuracy_score(y23_test,y23_predict))






classifier = GaussianNB()
classifier.fit(x25_train,y25_train)
y25_predict=classifier.predict(x25_test)
print("accuracy score",accuracy_score(y25_test,y25_predict))

classifier = RandomForestClassifier(n_estimators=60)
classifier.fit(x25_train,y25_train)
y25_predict=classifier.predict(x25_test)
print("accuracy score",accuracy_score(y25_test,y25_predict))

classifier = SVC(kernel='rbf',gamma='auto',C=2)
classifier.fit(x25_train,y25_train)
y25_predict=classifier.predict(x25_test)
print("accuracy score",accuracy_score(y25_test,y25_predict))

classifier = GradientBoostingClassifier()
classifier.fit(x25_train,y25_train)
y25_predict=classifier.predict(x25_test)
print("accuracy score",accuracy_score(y25_test,y25_predict))

classifier = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME')
classifier.fit(x25_train,y25_train)
y25_predict=classifier.predict(x25_test)
print("accuracy score",accuracy_score(y25_test,y25_predict))

classifier = LogisticRegression()
classifier.fit(x25_train,y25_train)
y25_predict=classifier.predict(x25_test)
print("accuracy score",accuracy_score(y25_test,y25_predict))

classifier = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100)
classifier.fit(x25_train,y25_train)
y25_predict=classifier.predict(x25_test)
print("accuracy score",accuracy_score(y25_test,y25_predict))





classifier = GaussianNB()
classifier.fit(x26_train,y26_train)
y26_predict=classifier.predict(x26_test)
print("accuracy score",accuracy_score(y26_test,y26_predict))

classifier = RandomForestClassifier(n_estimators=60)
classifier.fit(x26_train,y26_train)
y26_predict=classifier.predict(x26_test)
print("accuracy score",accuracy_score(y26_test,y26_predict))

classifier = SVC(kernel='rbf',gamma='auto',C=2)
classifier.fit(x26_train,y26_train)
y26_predict=classifier.predict(x26_test)
print("accuracy score",accuracy_score(y26_test,y26_predict))

classifier = GradientBoostingClassifier()
classifier.fit(x26_train,y26_train)
y26_predict=classifier.predict(x26_test)
print("accuracy score",accuracy_score(y26_test,y26_predict))

classifier = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME')
classifier.fit(x26_train,y26_train)
y26_predict=classifier.predict(x26_test)
print("accuracy score",accuracy_score(y26_test,y26_predict))

classifier = LogisticRegression()
classifier.fit(x26_train,y26_train)
y26_predict=classifier.predict(x26_test)
print("accuracy score",accuracy_score(y26_test,y26_predict))

classifier = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100)
classifier.fit(x26_train,y26_train)
y26_predict=classifier.predict(x26_test)
print("accuracy score",accuracy_score(y26_test,y26_predict))




# for F1 score of 4th descriptor
classifier = GaussianNB()
classifier.fit(x24_train,y24_train)
y24_predict=classifier.predict(x24_test)
print(classification_report(y24_test,y24_predict))

classifier =  RandomForestClassifier(n_estimators=60)
classifier.fit(x24_train,y24_train)
y24_predict=classifier.predict(x24_test)
print(classification_report(y24_test,y24_predict))

classifier = SVC(kernel='rbf',gamma='auto',C=2)
classifier.fit(x24_train,y24_train)
y24_predict=classifier.predict(x24_test)
print(classification_report(y24_test,y24_predict))

classifier = GradientBoostingClassifier()
classifier.fit(x24_train,y24_train)
y24_predict=classifier.predict(x24_test)
print(classification_report(y24_test,y24_predict))

classifier = AdaBoostClassifier(base_estimator=SVC(kernel='rbf',gamma='auto',C=2), algorithm='SAMME')
classifier.fit(x24_train,y24_train)
y24_predict=classifier.predict(x24_test)
print(classification_report(y24_test,y24_predict))

classifier = LogisticRegression()
classifier.fit(x24_train,y24_train)
y24_predict=classifier.predict(x24_test)
print(classification_report(y24_test,y24_predict))

classifier =  BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=100)
classifier.fit(x24_train,y24_train)
y24_predict=classifier.predict(x24_test)
print(classification_report(y24_test,y24_predict))







# for feature extraction but chi-square will not work here as some values are negative

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


bestfeatures=SelectKBest(score_func=chi2,k=100)
fit=bestfeatures.fit(X1,yCo1)

dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X1.columns)

featureScores=pd.contact([dfcolumns,dfscores],axis=1)
featureScores.columns=['Specs','Score']

featureScores



# for feature importance

from sklearn.ensemble import ExtraTreesClassifier
model= ExtraTreesClassifier()
model.fit(X1,yCo1)

print(model.feature_importances_)

from matplotlib.pyplot import figure
plt.rcParams['figure.figsize']=(10,12)

import matplotlib.pyplot as plt
feat_importances=pd.Series(model.feature_importances_,index=X1.columns)
feat_importances.nlargest(50).plot(kind='barh')
plt.show()

model.fit(X1,yCo2)
feat_importances=pd.Series(model.feature_importances_,index=X1.columns)
feat_importances.nlargest(50).plot(kind='barh')
plt.show()

model.fit(X2,yCo3)
feat_importances=pd.Series(model.feature_importances_,index=X2.columns)
feat_importances.nlargest(50).plot(kind='barh')
plt.show()

model.fit(X2,yCo4)
feat_importances=pd.Series(model.feature_importances_,index=X2.columns)
feat_importances.nlargest(50).plot(kind='barh')
plt.show()

model.fit(X2,yCo5)
feat_importances=pd.Series(model.feature_importances_,index=X2.columns)
feat_importances.nlargest(50).plot(kind='barh')
plt.show()

model.fit(X2,yCo6)
feat_importances=pd.Series(model.feature_importances_,index=X2.columns)
feat_importances.nlargest(50).plot(kind='barh')
plt.show()



'''


'''
pip install Ipython

import eli5
from eli5 import formatters
#from Ipython.display import display
#display(eli5.explain_weights(classifier,feature_names=list(features),importance_type='gain'))
#eli5.explain_weights(classifier,feature_names=list(features),importance_type='gain')
eli5.explain_weights(classifier,top=50)

pip install eli5

from eli5 import show_weights

show_weights(classifier, feature_names=X1.feature_names)

'''







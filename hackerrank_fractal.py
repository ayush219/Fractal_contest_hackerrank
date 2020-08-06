#Libraries
import numpy as np
import pandas as pd

#Dataset
train_dataset= pd.read_csv('train.csv', header=None)
X_train= train_dataset.iloc[:,:294].values
Y1_train= train_dataset.iloc[:,294].values
Y2_train= train_dataset.iloc[:,295].values
Y3_train= train_dataset.iloc[:,296].values
Y4_train= train_dataset.iloc[:,297].values
Y5_train= train_dataset.iloc[:,298].values
Y6_train= train_dataset.iloc[:,299].values

test_dataset=pd.read_csv('test.csv', header=None)
X_test= test_dataset.iloc[:,:294].values

#Classifier
from sklearn.linear_model import LogisticRegression
classifier1=LogisticRegression(random_state=0)
classifier1.fit(X_train,Y1_train)

classifier2=LogisticRegression( random_state=0)
classifier2.fit(X_train,Y2_train)

classifier3=LogisticRegression( random_state=0)
classifier3.fit(X_train,Y3_train)

classifier4=LogisticRegression( random_state=0)
classifier4.fit(X_train,Y4_train)

classifier5=LogisticRegression( random_state=0)
classifier5.fit(X_train,Y5_train)

classifier6=LogisticRegression( random_state=0)
classifier6.fit(X_train,Y6_train)

#Prediction
Y1_pred= classifier1.predict(X_test)
Y2_pred= classifier2.predict(X_test)
Y3_pred= classifier3.predict(X_test)
Y4_pred= classifier4.predict(X_test)
Y5_pred= classifier5.predict(X_test)
Y6_pred= classifier6.predict(X_test)

Y_pred=pd.DataFrame(list(zip(Y1_pred,Y2_pred,Y3_pred,Y4_pred,Y5_pred,Y6_pred)))

prediction = pd.DataFrame(Y_pred).to_csv('prediction.csv', index=None , header=None)
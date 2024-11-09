import joblib
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

data = sns.load_dataset('iris')
X = data.iloc[:,:-1]
Y = data.species
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.20,random_state=42)

log = LogisticRegression()
log.fit(x_train,y_train)
accuracy = accuracy_score(y_test, log.predict(x_test))
st.write('accuracy is',accuracy)
joblib.dump(log, 'iris_model.pkl')
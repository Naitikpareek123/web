import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

tel = pd.read_csv(r"C:\Users\91785\Documents\GitHub\web\project2.csv")
pd.set_option('display.max_rows',None)
print(tel.head(5))
print(tel.info())
print(tel.corr())
print(tel.describe())
print(tel['Churn'].value_counts())
print(tel.isnull().sum())
tel["Churn"]= tel["Churn"].map({"No": 0,"Yes": 1})
##print(tel['TelChurn'])
tel["PaperlessBilling"]= tel["PaperlessBilling"].map({"No": 0,"Yes": 1})
tel["StreamingMovies"]= tel["StreamingMovies"].map({"No": 0,"Yes": 1})
telx=['SeniorCitizen','tenure','PaperlessBilling','MonthlyCharges']
tely=['Churn']
print(tel.info())
x=tel[telx]
print(x)
y=tel[tely]
print(y)
print(x.head(5))
print(y.head(5))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x_train.head(5))
print(y_train.head(5))
Dec=DecisionTreeClassifier()
ram=RandomForestClassifier()
decision=Dec.fit(x_train,y_train)
pred = decision.predict(x_test)
# print(decision.score(x_test,pred))
print("Accuracy:",metrics.accuracy_score(y_test, pred))
ram=ram.fit(x_train,y_train)
r=ram.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,r))

import streamlit as st
st.title('Welcome To :violet[learn And Build]')

choice= st.selectbox('Login/Signup',['Login','Sign Up'])
if choice=='Login':
    emaill=st.text_input('Email Adddress')
    Password=st.text_input('Password',type='password')
    
    st.button('Login')

else:
    emaill=st.text_input('Email Adddress')
    First_Name=st.text_input('First Name')
    Last_Name=st.text_input('Last Name')
    Password=st.text_input('Password',type='password')
    st.button('Sign Up')
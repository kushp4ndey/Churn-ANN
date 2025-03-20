import os
os.chdir(r"C:\Users\admin\Desktop\Personal\GenAI Lectures\ANN Classification\annclassification")
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import streamlit as st

mdoel=tf.keras.models.load_model('model.h5')
model=load_model('model.h5')
with open('le_gender.pkl','rb') as file:
    le_gender=pickle.load(file)
with open('oh_geo.pkl','rb') as file:
    oh_geo=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

st.title('Customer Churn Prediction')
geography = st.selectbox('Geography',oh_geo.categories_[0])
gender = st.selectbox('Gender',le_gender.classes_)
age=st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary = st.number_input('Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_Cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data = ({
    'CreditScore': [credit_score],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_Cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
input_df = pd.DataFrame(input_data)
geo_encoded = oh_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=oh_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_df)
prediction=model.predict(input_data_scaled)
pred_proba = prediction[0][0]
st.write(f'Churn Probability: {pred_proba:.2f}')
if pred_proba>0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
import pandas as pd
import pickle

# Load the Trained Model
model = tf.keras.models.load_model("model.h5")

## Load the encoder and scaler
with open('label_encoder_gender.pkl' , 'rb') as file:
    label_encoder_gender = pickle.load(file=file)

with open('onehot_encoder_geo.pkl' , 'rb') as file:
    label_encoder_geo = pickle.load(file=file)

with open('scaler.pkl' , 'rb') as file:
    scaler = pickle.load(file)
    


## Streamlit App
st.title('Customer Churn Prediction')
geography = st.selectbox('Geography' , label_encoder_geo.categories_[0])
gender = st.selectbox('Geography' , label_encoder_gender.classes_)
age = st.slider('Age' , 18 , 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenure" , 0 , 10)
num_of_products = st.slider("Number of Products" , 1 ,4)
has_cr_card = st.selectbox('Has Credit Card' ,[0,1])
is_active_member = st.selectbox('Is Active Member' ,[0,1])


## Input Data
input_data = {
    'CreditScore' : credit_score,
    'Geography' : geography,
    'Gender' : gender,
    'Age' : age,
    'Tenure' : tenure,
    'Balance' : balance,
    'NumOfProducts' : num_of_products,
    'HasCrCard' : has_cr_card ,
    'IsActiveMember' : is_active_member,
    'EstimatedSalary' : estimated_salary
}

data = pd.DataFrame([input_data])

## Encode the Geogarphy

geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded , columns=label_encoder_geo.get_feature_names_out(['Geography']))

data['Gender'] = label_encoder_gender.transform(data['Gender'])

data = pd.concat([data.drop('Geography' , axis=1) , geo_encoded_df] ,axis=1)

## Scaling the Input data
input_scaled = scaler.transform(data)

## Predict Churn
prediction = model.predict(input_scaled)
prediction_probab = prediction[0][0]

st.write(f'Churn Probablity : {prediction_probab:.2f}')

if prediction_probab > 0.5 :
    st.write("The Customer is Likely to Churn")
else:
    st.write("The Customer is not Likely to Churn")
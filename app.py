import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Stroke Prediction App')

st.write("""
This app predicts the likelihood of a stroke based on various health parameters.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 0, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension', ('No', 'Yes'))
    heart_disease = st.sidebar.selectbox('Heart Disease', ('No', 'Yes'))
    ever_married = st.sidebar.selectbox('Ever Married', ('No', 'Yes'))
    work_type = st.sidebar.selectbox('Work Type', ('Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'))
    Residence_type = st.sidebar.selectbox('Residence Type', ('Urban', 'Rural'))
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 25.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', ('never smoked', 'formerly smoked', 'smokes'))

    data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [1 if hypertension == 'Yes' else 0],
        'heart_disease': [1 if heart_disease == 'Yes' else 0],
        'ever_married': [1 if ever_married == 'Yes' else 0],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    }

    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Dummy encode categorical variables
input_df = pd.get_dummies(input_df, drop_first=True)

# Ensure input data matches the model's expected data structure
expected_columns = ['gender','age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                    'avg_glucose_level', 'bmi', 'smoking_status']

# Add missing columns with default values
missing_cols = set(expected_columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# Ensure correct column order
input_df = input_df[expected_columns]

st.subheader('User Input Parameters')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('Stroke' if prediction[0] == 1 else 'No Stroke')

st.subheader('Prediction Probability')
st.write(prediction_proba)

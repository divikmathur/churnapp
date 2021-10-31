# Originally created: 02nd October, 2021
#                     Gandhi Jayanti
# Last amended: 11th Oct, 2021
# Myfolder: /home/ashok/Documents/churnapp
#           VM: lubuntu_healthcare
#           D:\data\OneDrive\Documents\streamlit
# Ref: https://builtin.com/machine-learning/streamlit-tutorial
#
# Objective:
#             Deploy an ML model on web
#
########################
# Notes:
#       1, Run this app in its folder, as:
#          cd /home/ashok/Documents/churnapp
#          streamlit  run  churn-app.py
#       2. Accomanying file to experiment is
#          expt.py
########################

# 1.0 Call libraries
# Install as: pip install streamlit
# Better create a separate conda environment for it
import streamlit as st
import pandas as pd
import numpy as np
import pickle



# 1.1 Our data columns:
    
data_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
             'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
             'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
             'MonthlyCharges', 'TotalCharges']




# 1.2 Write some body-text on the Web-Page:

st.write("""
# Churn Prediction App

Customer churn is defined as the loss of customers after a certain period of time.
Companies are interested in targeting customers who are likely to churn. They can
target these customers with special deals and promotions to influence them to stay
with the company.

This app predicts the probability of a customer churning using Telco Customer data. Here
customer churn means the customer does not make another purchase after a period of time.
""")




# 2.0 Create a component to upload data file in the sidebar.
#     'uploaded_file' is a pandas dataframe

uploaded_file = st.sidebar.file_uploader(
                                          "Upload your input CSV file",
                                           type=["csv"]
                                         )



# 2.1 If no file is uploaded, here is a data entry form
#     to capture some feature values:

def user_input_features():
    # 2.1.1  Create four widgets
    gender         = st.sidebar.selectbox('gender',('Male','Female'))
    PaymentMethod  = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0,118.0, 18.0)
    tenure         = st.sidebar.slider('tenure', 0.0,72.0, 0.0)
    # 2.1.2 Collect widget-output in a dictionary
    feature_dict = {
                    'gender':        [gender],         # Should be a list data structure
                    'PaymentMethod': [PaymentMethod],
                    'MonthlyCharges':[MonthlyCharges],
                    'tenure':        [tenure]
                    }
    # 2.1.3 Return a dictionary of feature values
    return feature_dict



# 3.0 Get data either from an uploaded file
#     or read values from widgets:
    
if uploaded_file is not None:
    # 3.1 Read the uploaded file
    input_df = pd.read_csv(uploaded_file)
    customerid = input_df['customerID']
    input_df = input_df.iloc[:,1:]    # Exclude customerID
else:
    # 3.2 Create a DataFrame with only np.nan values
    input_df = pd.DataFrame( [tuple([np.nan] * len(data_cols))] , columns = data_cols )
    # 3.3 Read values from widgets
    feature_dict = user_input_features()
    # 3.4 Amend input_df 
    input_df['gender'] = feature_dict['gender']
    input_df['PaymentMethod'] = feature_dict['PaymentMethod']
    input_df['MonthlyCharges'] = feature_dict['MonthlyCharges']
    input_df['tenure'] = feature_dict['tenure']
    

# 4.0
if uploaded_file is not None:
    # 4.0.1 Write the first row
    st.write("CustomerID:", customerid)
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)



# 5.0 Read in saved classification model
#     from the current folder:
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))

# 5.1 Apply model to make predictions
prediction = load_clf.predict(input_df)               # 'prediction' will have value of 1 or 0
prediction_proba = load_clf.predict_proba(input_df)   # Prediction probability


# 5.2 Display Labels
st.subheader('Prediction')
churn_labels = ['No','Yes']          # churn_labels is a list of strings
                                     # churn_labels[0] is 'No' and churn_labels[1] is 'Yes'

# 5.2.1
st.write(churn_labels[prediction[0]])   # prediction is an array while prediction[0] is scalar

# 5.3 Also display probabilities
st.subheader('Prediction Probability')
# 5.3.1 Numpy arrays are displayed with column names
#     as 1 or 0
st.write(prediction_proba)
######################################

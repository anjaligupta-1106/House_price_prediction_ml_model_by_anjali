import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
import pickle
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

st.title('Housing Price Prediction Using Machine Learning')

st.image('https://www.archistar.ai/wp-content/uploads/2022/03/Determining-Future-House-Prices-And-Their-Impact-on-Property-Development.jpg')

st.write('''Our project, Housing Price Prediction Using Machine Learning, aims to predict house prices based on various factors such as location, property size, number of bedrooms, and other important features. By using machine learning algorithms, we analyze housing data and generate price predictions. This project helps buyers, sellers, and real estate professionals make better decisions through data-driven insights and efficient property price estimation.''')


data = fetch_california_housing()
df = pd.DataFrame(data.data,columns = data.feature_names)
df['MedHouseVal'] = data.target

X = df.iloc[:, :-2]
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)

st.sidebar.title('Select House Features🏠')
st.sidebar.image('https://i.pinimg.com/originals/46/cb/a9/46cba9f93a3d437db6d42f4bcd1a5f5f.gif')

all_value = []
for i in X.columns:
    value = st.sidebar.slider(f'Select {i} value')
    all_value.append(value)

final_data = [all_value]
final_data = scaler.transform(final_data)

with open('chatgpt_for_house.pkl', 'rb') as f:
    chatgpt = pickle.load(f)
print('Model Loaded Successfully!!!')

price = chatgpt.predict(final_data)[0]

if price:
    with st.spinner('Wait for price prediction...'):
        time.sleep(5)
    st.success(f'Predicted price is: $ {np.abs(round(price*100000, 2))}')
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
from pickle import load


# In[3]:


scaler = load(open('standard_scaler.pkl', 'rb'))
knn_regressor = load(open('knn_model.pkl', 'rb'))

st.set_page_config(page_title="Hello", page_icon="👋")
st.title(" 💎 *'Predict the Diamond Price!'* 🔮 ")
st.header(" ")
st.subheader("Enter all the features of the Diamond:")


# In[4]:


carat = st.slider("Carat", 0.00, 6.00)
length = st.slider("Length", 0.00, 15.00)
width = st.slider("Width", 0.00, 65.00)
depth = st.slider("Depth", 0.00, 35.00)
cut = st.selectbox("Cut", ("Ideal", "Premium", "Very Good", "Good", "Fair"))
color = st.selectbox("Color", ("D", "E", "F", "G", "H", "I","J"))
clarity = st.selectbox("Clarity", ("I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"))

btn_click = st.button("Predict Price")


# In[5]:


if btn_click == True:
    if carat and cut and color and clarity and length and width and depth:
        label_cut = {'Ideal':2, 'Premium':3, 'Very Good':4, 'Good':1, 'Fair':0}
        label_color = {'G':3, 'E':1, 'F':2, 'H':4, 'D':0, 'I':5, 'J':6}
        label_clarity = {'SI1':2, 'VS2':5, 'SI2':3, 'VS1': 4, 'VVS2':7, 'VVS1':6, 'IF':1, 'I1':0}

        cut_labelled = label_cut[cut]
        color_labelled = label_color[color]
        clarity_labelled = label_clarity[clarity]

        query_point = np.array([float(carat), float(cut_labelled), float(color_labelled), float(clarity_labelled), float(length), float(width), float(depth)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = knn_regressor.predict(query_point_transformed)
        st.success(pred)
        st.balloons()
    else:
        st.error("Enter the values properly!")
        st.snow()


# In[6]:


page_bg_img = '''
<style>
.stApp {
background-image: url("https://th.bing.com/th/id/R.233b7cf019442dc0e13c077a584e7129?rik=ehFAK8EN%2bc%2f%2fbw&riu=http%3a%2f%2fpapers.co%2fwallpaper%2fpapers.co-mh49-diamond-dark-two-art-35-3840x2160-4k-wallpaper.jpg&ehk=7poMx0SpJtB8x8fLLSWZqzIa6XtARJJ%2b93c6%2bTzyPaw%3d&risl=&pid=ImgRaw&r=0.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


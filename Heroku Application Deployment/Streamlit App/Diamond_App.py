#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pickle import load


################################## DATAFRAME CODES #######################################
df = pd.read_csv("diamonds.csv")
df.rename(columns={'x':'length', 'y':'width', 'z':'depth', 'depth':'depth%'}, inplace = True)
df_target = df[['price']]
df = df.drop('price', axis = 1)
df = pd.concat([df, df_target], axis = 1)
df[['length','width','depth']] = df[['length','width','depth']].replace(0, np.NaN)
df.dropna(inplace = True)
corr = df.corr()
fig = plt.figure()
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(corr , xticklabels=corr.columns , yticklabels=corr.columns , annot=True)

# Splitting the target and independent columns
X = df[['carat', 'cut', 'color', 'clarity', 'length', 'width', 'depth']]
y = df[['price']]

# Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 4)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
# label encoding the train setcategorical columns
le = LabelEncoder()
X_train['cut']=le.fit_transform(X_train['cut'])
X_train['color']=le.fit_transform(X_train['color'])
X_train['clarity']=le.fit_transform(X_train['clarity'])
# label encoding the test set categorical columns
le= LabelEncoder()
X_test['cut']=le.fit_transform(X_test['cut'])
X_test['color']=le.fit_transform(X_test['color'])
X_test['clarity']=le.fit_transform(X_test['clarity'])

# Standardization
# Data preprocessing on training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_rescaled = pd.DataFrame(scaler.fit_transform(X_train),
                                columns = X_train.columns,
                                index = X_train.index)

# Data preprocessing on testing data
X_test_rescaled = pd.DataFrame(scaler.transform(X_test),
                               columns = X_test.columns,
                               index = X_test.index)


################################## STREAMLIT CODES #######################################
st.set_page_config(page_title="Hello", page_icon="👋")
st.title(" 💎 *'Diamond Price Prediction!'* 💎 ")
st.header("*'Be like a diamond precious and rare not like a stone found everywhere!'*")
st.subheader(" ")
st.subheader("Data Preview")

# Preview of the dataset
if(st.checkbox("Data Head/Tail")):
    preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
    if(preview == "Top"):
        st.write(df.head())
    if(preview == "Bottom"):
        st.write(df.tail())
    st.write("Our Target Variable is Price in this dataset.")

# display the whole dataset
if(st.checkbox("Show complete Dataset")):
    st.write(df)

# Show shape
if(st.checkbox("Display the shape")):
    st.write(df.shape)
    dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
    if(dim == "Rows"):
        st.write("Number of Rows", df.shape[0])
    if(dim == "Columns"):
        st.write("Number of Columns", df.shape[1])

# show columns
if(st.checkbox("Show the Columns")):
    st.write(df.columns)
    
# correlation between features
if(st.checkbox("Correlation between features")):
    st.pyplot(fig)
    st.write("We observe that Carat, Length, Width and Depth are highly correlated with price. Therefore, we won't be using Depth% and Table to predict the price.")


################################## BACKGROUND CODES #######################################
page_bg_img = '''
<style>
.stApp {
background-image: url("https://th.bing.com/th/id/R.233b7cf019442dc0e13c077a584e7129?rik=ehFAK8EN%2bc%2f%2fbw&riu=http%3a%2f%2fpapers.co%2fwallpaper%2fpapers.co-mh49-diamond-dark-two-art-35-3840x2160-4k-wallpaper.jpg&ehk=7poMx0SpJtB8x8fLLSWZqzIa6XtARJJ%2b93c6%2bTzyPaw%3d&risl=&pid=ImgRaw&r=0.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


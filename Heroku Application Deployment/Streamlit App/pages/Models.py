#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[2]:


################################## STREAMLIT CODES #######################################
st.set_page_config(page_title="Models", page_icon="👋")
st.title(" 💎 *'Models Evaluation'* 🔬 ")
st.header(" ")


# In[5]:


# Linear Regression Model
if(st.checkbox("Linear Regression Model")):
    lr_model = '''
    #Importing LinearRegression from sklearn.linear_model module
    from sklearn.linear_model import LinearRegression
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train_rescaled, y_train))
    
    # Make prediction on test set
    y_test_pred_lr = linear_regressor.predict(X_test_rescaled)
    
    # Calculate Accuracy of predictions
    from sklearn import metrics
    print("R2 Score: " , metrics.r2_score(y_test, y_test_pred_lr))
    print("Root Mean Squared Error: " , np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_lr)))
    '''
    
    st.code(lr_model)
    st.subheader("R2 Score:  0.89")
    st.subheader("Root Mean Squared Error:  1328.66")


# In[7]:


# KNN Regression Model
if(st.checkbox("KNN Regression Model")):
    knn_model = '''
    #Importing KNN regression from sklearn.neighbours module
    from sklearn.neighbors import KNeighborsRegressor
    knn_regressor = KNeighborsRegressor()
    knn_regressor.fit(X_train_rescaled, y_train)
    # Make prediction on test set
    y_test_pred_knn = knn_regressor.predict(X_test_rescaled)
    # Calculate Accuracy of predictions
    from sklearn import metrics
    print("R2 Score: " , metrics.r2_score(y_test, y_test_pred_knn))
    print("Root Mean Squared Error: " , np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_knn)))
    '''
    st.code(knn_model)
    st.subheader("R2 Score:  0.97")
    st.subheader("Root Mean Squared Error:  622.77")


# In[8]:


# Decision Tree Regression Model
if(st.checkbox("Decision Tree Regression Model")):
    dt_model = '''
    # Importing Decision Tree Regression from sklearn.tree module
    from sklearn.tree import DecisionTreeRegressor
    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(X_train_rescaled, y_train)
    # Make prediction on test set
    y_test_pred_dt = dt_regressor.predict(X_test_rescaled)
    # Calculate Accuracy of predictions
    from sklearn import metrics
    print("R2 Score: " , metrics.r2_score(y_test, y_test_pred_dt))
    print("Root Mean Squared Error: " , np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_dt)))
    '''
    st.code(dt_model)
    st.subheader("R2 Score:  0.97")
    st.subheader("Root Mean Squared Error:  730.81")


# In[9]:


# Random Forest Regression Model
if(st.checkbox("Random Forest Regression Model")):
    rf_model = '''
    # Importing Random Forest Regression from sklearn.ensemble module
    from sklearn.ensemble import RandomForestRegressor
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train_rescaled, y_train)
    # Make prediction on test set
    y_test_pred_rf = rf_regressor.predict(X_test_rescaled)
    # Calculate Accuracy of predictions
    from sklearn import metrics
    print("R2 Score: ", metrics.r2_score(y_test, y_test_pred_rf))
    print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_rf)))
    '''
    st.code(rf_model)
    st.subheader("R2 Score:  0.98")
    st.subheader("Root Mean Squared Error:  542.69")


# In[10]:


if(st.checkbox("Model Selected for Prediction")):
    st.subheader("We observed that KNN Regression Model gives the best evaluation score compared to the rest of the models. Therefore we will use KNN Regression Model to predict the price of the Diamond!💎")


# In[11]:


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


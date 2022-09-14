#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle

model = pickle.load(open('random.pkl','rb'))

def welcome():
    return 'welcome all'


def prediction(carat,cut,color,clarity,x,y,z,depth,table):
    prediction = model.predict(
        [[carat,cut,color,clarity,x,y,z,depth,table]])
    print(prediction)
    return prediction


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Predict the diamond price")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:black;padding:13px">
    <h1 style ="color:white;text-align:center;"> Predicting the diamond price ML App </h1>
    </div>
    <h2> Enter the details about your diamond to know its price </h2>
    
    """

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    carat = st.text_input("Weight of the diamond")
    cut = st.text_input("Quality of the cut on  diamond")
    color = st.text_input("Color of the diamond")
    clarity = st.text_input("Clarity of the diamond")
    x = st.text_input("Length of diamond in mm")
    y = st.text_input("Width of diamond in mm")
    z = st.text_input("Depth of diamond in mm")
    depth = st.text_input("Total depth percentage")
    table = st.text_input("width of top of diamond relative to widest point")
    result = ""

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(carat,cut,color,clarity,x,y,z,depth,table)
    st.success('The price of your diamond is  {}'.format(result))


if __name__ == '__main__':
    main()


# In[ ]:





import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Nearest Pub"
)

df = pd.read_csv("open_pubs_updated.csv")

# make header
st.header("Nearest 5 Pubs from your Location : 🍾🍸🍾")


lat = st.number_input('The Latitude of Place')
lon = st.number_input('The Longitude of Place')
button = st.button("Submit")
df_new = df[['latitude', 'longitude']]
new_points = np.array([lat, lon])
# Calculate distance between new_points and all points in df_new
distances = np.sqrt(np.sum((new_points - df_new)**2, axis = 1))


# sort the array using arg partition and keep n elements
n = 5
min_indices = np.argpartition(distances,n-1)[:n]
if button:
    st.text("The location corresponsing to below minimum distances : ")
    st.dataframe(df.iloc[min_indices])
    st.text("The minimum distances are:")
    st.dataframe(distances.head(5))
    st.map(df.iloc[min_indices])

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

st.write("""
# AIRBNB HOUSE PRICE PREDICT APP!
""")
image=Image.open('Hairb.jpg')
st.image(image, caption='Home away from Home')


model=pickle.load(open('model(xg).pkl','rb'))

scaler=pickle.load(open('scaler.pkl','rb'))


st.sidebar.header('User Input Parameters')

def user_input_features():
    Room_type=st.sidebar.selectbox('Room Type',('Private room','Entire home/apt','Shared room'))
    if Room_type=='Entire home/apt':
        Entire=1
        Private=0
        Shared=0
    
    if Room_type=='Private room':
        Entire=0
        Private=1
        Shared=0
        
    if Room_type=='Shared room':
        Entire=0
        Private=0
        Shared=1
        
    Region_hood=st.sidebar.selectbox('Region',('North Region','Central Region','East Region','West Region','North_East Region')) 
    if Region_hood=='Central Region':
        Central=1
        North=0
        East=0
        West=0
        North_East=0
        
    if Region_hood=='East Region':
        Central=0
        North=0
        East=1
        West=0
        North_East=0
        
    if Region_hood=='North Region':
        Central=0
        North=1
        East=0
        West=0
        North_East=0
        
    if Region_hood=='West Region':
        Central=0
        North=0
        East=0
        West=1
        North_East=0
        
    if Region_hood=='North_East Region':
        Central=0
        North=0
        East=0
        West=0
        North_East=1
        


    Host_id=st.number_input('What is the Host ID')
    host_list_count=st.number_input('Host Listing Count')
    longitude=st.number_input("Building's longitudinal location")
    latitude=st.number_input("Building's latitudinal location")
    minimum_nights=st.number_input('How many nights will you be staying for')
    availability=st.number_input('How many days is the building available for')
    last_rev_month=st.number_input('On which month was the last review',max_value=12,min_value=1,step=1)
    last_rev_year=st.number_input('On which year was the last review',max_value=2022,min_value=2012,step=1)
    no_reviews=st.number_input('Number of reviews recieved')
    reviews_per_month=st.number_input('How many reviews per month')


    
    
    data = {'latitude':latitude,
            'host_id':Host_id,
            'longitude': longitude ,
            'minimum_nights':minimum_nights,
            'calculated_host_listings_count':host_list_count,
            'availability_365':availability,
            'last_review_month':last_rev_month,
            'last_review_year':last_rev_year,
            'number_of_reviews':no_reviews,
            'reviews_per_month':reviews_per_month,
            'room_type_Private room':Private,
            'room_type_Shared room':Shared,
            'neighbourhood_group_North-East Region':North_East,
            'neighbourhood_group_East Region':East,
            'neighbourhood_group_North Region':North,
            'neighbourhood_group_West Region':West
            }

    features=pd.DataFrame(data,index=[0])
    return features
    
input_df=user_input_features()
input_df=scaler.transform(input_df)

if st.button('PREDICT'):
    y_out=model.predict(input_df)
    st.write(f'This room will cost you $' ,int(y_out[0]))







        
       
    



    

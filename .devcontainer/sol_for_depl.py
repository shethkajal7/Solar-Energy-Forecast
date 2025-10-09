#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from prophet import Prophet
from pylab import rcParams
import pickle
import streamlit as st
import matplotlib.pyplot as plt

with open('solar_energy_forecast.pkl','rb') as file:
    model = pickle.load(file)
    

st.title("1.5MWp_Solar Energy Forecast")
st.write("This app provides yield expectation charts for today and tomorrow for a solar plant")
st.subheader("Enter Forecasting Details")

days = st.number_input("Forecasting Days", min_value=0, max_value=800)


future1 = model.make_future_dataframe(periods=days,freq='D')


if st.button("Forecast"):
    forecast1 = model.predict(future1)
    # Plot the forecast
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    fig = model.plot(forecast1,ax=ax)
    st.pyplot(fig)
    
    


# In[7]:





# In[ ]:




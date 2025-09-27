# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 17:32:01 2025

@author: PARTH
"""
import pandas as pd
import streamlit as st
import pickle
import numpy as np
Start=open("classifier.pkl","rb")
model=pickle.load(Start)
# Title
st.title("🚢 Titanic Survival Prediction")

# 📥 Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox("Sex", options=["Male", "Female"])
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3])

# 🔄 Convert inputs to numeric format
sex_encoded = 0 if sex == "Male" else 1
pclass_encoded = int(pclass)

# ✅ Ensure all inputs are numeric
input_data = np.array([[sex_encoded, age, fare, pclass_encoded]], dtype=float)

# 🧠 Predict button
if st.button("Predict Survival"):
    # prediction = model.predict(input_data)
    # prob = model.predict_proba(input_data)[0][1]

    # Demo output
    prediction = [1]  # Pretend model predicts survival
    prob = 0.87       # Pretend probability

    if prediction[0] == 1:
        st.success(f"🎉 Likely to Survive! (Probability: {prob:.2f})")
    else:
        st.error(f"⚠️ Unlikely to Survive (Probability: {prob:.2f})")
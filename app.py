import streamlit as st
import numpy as np
import pickle

# Load models and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Title and Description
st.title('Crop Recommendation System 🌱')
st.write('This is a crop recommendation system based on machine learning. Enter the agricultural data to get the best crop recommendation.')

# Input fields with styled boxes and rounded corners
Nitrogen = st.number_input('Enter Nitrogen', min_value=0.0, step=0.01)
Phosphorus = st.number_input('Enter Phosphorus', min_value=0.0, step=0.01)
Potassium = st.number_input('Enter Potassium', min_value=0.0, step=0.01)
Temperature = st.number_input('Enter Temperature in °C', min_value=-50.0, step=0.1)
Humidity = st.number_input('Enter Humidity in %', min_value=0.0, step=0.1)
pH = st.number_input('Enter pH value', min_value=0.0, step=0.01)
Rainfall = st.number_input('Enter Rainfall in mm', min_value=0.0, step=0.1)

# Prediction logic
if st.button('Get Recommendation'):
    feature_list = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        st.markdown(f"""
            <div style="background-color:#2c7be5; color:white; padding:20px; border-radius:10px; font-size:40px; text-align:center; 
                        animation: fadeIn 1s;">
                The best crop to cultivate is: {crop} 🌾
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Sorry, we could not determine the best crop to be cultivated with the provided data.")

# Adding background styling, animations, and responsive layout
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f0f0f5 50%, #d1e7ff);
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #ff7f50;
            color: white;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 30px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #ff6347;
            transform: scale(1.1);
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 10px;
            border: 2px solid #ccc;
        }
        .stTextInput>div>div>input:focus {
            border-color: #2c7be5;
            outline: none;
        }
        .stButton>button:focus {
            outline: none;
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
    </style>
""", unsafe_allow_html=True)








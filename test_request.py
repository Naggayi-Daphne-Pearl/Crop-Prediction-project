import requests
import json
import numpy as np

# Function URL from your deployment
url = "https://us-central1-knowledge-distillation-452212.cloudfunctions.net/predict-handler-v2"

def create_test_data():
    # Create a realistic sample (these values should be scaled/encoded)
    features = [
        1.0,    # Station (e.g., Arua encoded)
        0.5,    # Latitude (scaled)
        0.3,    # Longitude (scaled)
        0.7,    # Elevation (scaled)
        0.6,    # Temperature (scaled)
        0.4,    # Rainfall (scaled)
        0.5,    # Humidity (scaled)
        0.6,    # Solar Radiation (scaled)
        0.3,    # Wind Speed (scaled)
        2.0,    # Soil Type (encoded)
        0.5,    # Soil pH (scaled)
        0.6,    # Soil Moisture (scaled)
        0.5,    # Soil Temperature (scaled)
        0.4,    # Nitrogen Content (scaled)
        0.3,    # Phosphorus Content (scaled)
        0.4,    # Potassium Content (scaled)
        0.5,    # Organic Matter (scaled)
        1.0,    # Season (encoded)
        3.0,    # Month (encoded)
        0.8,    # Year (scaled)
        0.0     # Previous Crop (encoded)
    ]
    
    return features

try:
    features = create_test_data()
    
    test_data = {
        "features": features
    }
    
    print("\nSending features:")
    for i, value in enumerate(features):
        print(f"Feature {i+1}: {value}")
    print("Number of features:", len(features))
    
    response = requests.post(url, json=test_data)
    
    print("\nStatus Code:", response.status_code)
    if response.status_code == 200:
        result = response.json()
        print("Prediction:", result['prediction'])
        print("Confidence:", result['confidence'])
    else:
        print("Error:", response.text)
        print("Response Headers:", response.headers)
except Exception as e:
    print("Error:", str(e)) 
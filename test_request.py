import requests
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function URL from your deployment
url = "https://us-central1-knowledge-distillation-452212.cloudfunctions.net/predict-handler-v2"

# Load and prepare test data
def prepare_test_data():
    # Load the CSV file
    df = pd.read_csv("GygaUganda - Station.csv")
    
    # Get one sample row (excluding the CROP column)
    sample = df.drop(columns=["CROP"]).iloc[0]
    
    # Convert to list and ensure all values are float
    features = [float(x) for x in sample.values]
    
    return features

# Prepare the test data
features = prepare_test_data()

# Create the request payload
test_data = {
    "features": features
}

print("Sending features:", features)

# Make the request
try:
    response = requests.post(url, json=test_data)
    
    # Print results
    print("\nStatus Code:", response.status_code)
    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Error:", response.text)
except Exception as e:
    print("Error making request:", e) 
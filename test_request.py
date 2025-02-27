import requests
import json
import numpy as np

# Function URL from your deployment
url = "https://us-central1-knowledge-distillation-452212.cloudfunctions.net/predict-handler-v2"

def create_test_data():
    # Create a sample of preprocessed features
    # These should be already scaled numerical values and encoded categorical values
    features = np.zeros(21)  # Adjust this number to match your model's input dimension
    
    # You can adjust these values based on your actual data distribution
    features = [
        0.0, 0.0, 0.0, 0.0, 0.0,  # First 5 features
        0.0, 0.0, 0.0, 0.0, 0.0,  # Next 5 features
        0.0, 0.0, 0.0, 0.0, 0.0,  # Next 5 features
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Last 6 features
    ]
    
    return features

try:
    # Get test features
    features = create_test_data()
    
    # Create the request payload
    test_data = {
        "features": features
    }
    
    print("\nSending features:", features)
    print("Number of features:", len(features))
    
    # Make the request
    response = requests.post(url, json=test_data)
    
    # Print results
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
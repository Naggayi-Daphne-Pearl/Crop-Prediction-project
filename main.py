import functions_framework
from flask import jsonify
import torch
import numpy as np
from model2_ed import StudentModel
import json

# Load model and other necessary files
def load_model():
    try:
        # Load model architecture info
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Initialize model
        model = StudentModel(
            input_dim=model_info['input_dim'],
            output_dim=model_info['output_dim']
        )
        
        # Load trained weights
        model.load_state_dict(torch.load('student_model.pth', map_location=torch.device('cpu')))
        model.eval()
        
        # Load class info
        with open('class_info.json', 'r') as f:
            class_info = json.load(f)
        
        return model, class_info
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Initialize model globally
try:
    model, class_info = load_model()
except Exception as e:
    print(f"Error initializing model: {e}")
    raise

@functions_framework.http
def predict_handler(request):
    """HTTP Cloud Function."""
    try:
        # Set CORS headers for the preflight request
        if request.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '3600'
            }
            return ('', 204, headers)

        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }

        # Get the request JSON
        request_json = request.get_json()
        if not request_json:
            return jsonify({'error': 'No input data provided'}), 400, headers

        # Convert input to tensor
        input_data = torch.tensor(request_json['features'], dtype=torch.float32)
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_data)
            _, predicted = torch.max(outputs, 1)
            
            # Convert prediction to crop name
            predicted_crop = class_info['class_names'][predicted.item()]
        
        # Return prediction
        return jsonify({
            'prediction': predicted_crop,
            'confidence': float(outputs.max().item())
        }), 200, headers

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500, headers
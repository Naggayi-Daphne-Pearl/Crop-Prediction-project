import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle 
import json

# Student Model Definition
class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load the Trained Model
def load_model(model_path, input_dim, output_dim):
    try:
        model = StudentModel(input_dim, output_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please ensure you have run model2-ed.py to train and save the model first.")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Preprocessing Functions
def preprocess_data(df, numerical_cols, categorical_cols, scaler):
    # Fill missing values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Encode categorical columns using saved LabelEncoders
    for col in categorical_cols:
        with open(f'{col}_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        df[col] = encoder.transform(df[col].astype(str))

    # Normalize numerical columns
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    return df

# Prediction Function
def predict_crop(data, model, numerical_cols, categorical_cols, scaler):
    df = pd.DataFrame([data], columns=numerical_cols + categorical_cols)
    df = preprocess_data(df, numerical_cols, categorical_cols, scaler)
    input_tensor = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
        _, predicted_class_idx = torch.max(prediction, 1)
        predicted_class_name = class_names[predicted_class_idx.item()]
    return predicted_class_name

# Load column information
with open('column_info.json', 'r') as f:
    column_info = json.load(f)
    numerical_cols = column_info['numerical_cols']
    categorical_cols = column_info['categorical_cols']

# Load encoder dictionary
with open('encoder_dict.json', 'r') as f:
    encoder_dict = json.load(f)

# Initialize and set up scaler
scaler = StandardScaler()
try:
    # Load scaler parameters
    scaler.mean_ = np.load('scaler_mean.npy')
    scaler.var_ = np.load('scaler_var_.npy')
    scaler.scale_ = np.sqrt(scaler.var_)
except (FileNotFoundError, EOFError) as e:
    print("Error loading scaler parameters:", e)
    print("Please ensure model2-ed.py has been run to generate the scaler files")
    exit(1)

print("Successfully loaded preprocessing parameters:")
print(f"Number of numerical columns: {len(numerical_cols)}")
print(f"Number of categorical columns: {len(categorical_cols)}")
print(f"Scaler mean shape: {scaler.mean_.shape}")
print(f"Scaler variance shape: {scaler.var_.shape}")

# Example Usage (replace with your actual paths and data)
input_dim = len(numerical_cols) + len(categorical_cols)

# Load class information instead of using encoder length
with open('class_info.json', 'r') as f:
    class_info = json.load(f)
    output_dim = class_info['n_classes']
    class_names = class_info['class_names']

print(f"Number of classes: {output_dim}")
print("Class names:", class_names)

# Initialize model with correct dimensions
model_path = 'student_model.pth'
model = load_model(model_path, input_dim, output_dim)

#Example of how to call the predict function.
#data = [1,2,3,'category1','category2'] #fill in data.
#prediction = predict_crop(data, model, numerical_cols, categorical_cols, scaler, encoder_dict)
#print(prediction)
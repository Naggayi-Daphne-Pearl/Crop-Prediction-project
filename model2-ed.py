#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json  # Add this import
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import torch.nn.functional as F  # Add this import if not already present
from google.oauth2 import service_account
from googleapiclient.discovery import build
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



try:
    # Try different possible file names
    possible_files = [
        'GygaUganda - Station.csv',
        'Station.csv',
        'GygaModelRunsUganda.csv'
    ]
    
    df = None
    for file_name in possible_files:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            print(f"Successfully loaded {file_name}")
            break
    
    if df is None:
        raise FileNotFoundError("Could not find the input CSV file. Please ensure one of these files exists: " + ", ".join(possible_files))

except Exception as e:
    print(f"Error loading data: {e}")
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir())
    raise


# In[5]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# Prepare X and y
X = df.drop(columns=["CROP"])  
y = df["CROP"]

# Check unique values of target
print("Unique crop types:", df["CROP"].unique())
print("Crop counts:\n", df['CROP'].value_counts())

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Fill missing values BEFORE scaling
X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

# Initialize encoders
encoder_dict = {}
encoders = {}  # Dictionary to store encoder objects

# Encode categorical columns properly
for col in categorical_cols:
    # Create and fit a new encoder for each column
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col].astype(str))
    
    # Convert numpy.int64 to regular Python int when creating the dictionary
    encoder_dict[col] = {str(category): int(index) for category, index in 
                        zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_))}
    
    # Save the encoder object
    with open(f'{col}_encoder.pkl', 'wb') as f:
        pickle.dump(encoders[col], f)
    
    print(f"Encoding for {col}: {encoder_dict[col]}")

# Encode target variable if categorical
if y.dtype == 'object' or isinstance(y.iloc[0], str):
    crop_encoder = LabelEncoder()
    y = crop_encoder.fit_transform(y)
    
    # Save the CROP encoder
    with open('CROP_encoder.pkl', 'wb') as f:
        pickle.dump(crop_encoder, f)

# Normalize numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save scaler parameters for deployment
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_var_.npy', scaler.var_)

# Print scaler parameters to verify
print("\nScaler mean values:", scaler.mean_)
print("\nScaler variance values:", scaler.var_)

# Save encoder dictionary and column info with error handling
try:
    # Save encoder dictionary
    with open('encoder_dict.json', 'w') as f:
        json.dump(encoder_dict, f, indent=4)
    print("Successfully saved encoder dictionary")

    # Save column information
    column_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }
    with open('column_info.json', 'w') as f:
        json.dump(column_info, f, indent=4)
    print("Successfully saved column information")

except Exception as e:
    print(f"Error saving JSON files: {e}")
    print("Encoder dictionary:", encoder_dict)
    raise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# After the train_test_split and before printing tensor shapes
# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Now we can print the tensor shapes
print("Tensor shapes:")
print("X_train_tensor:", X_train_tensor.shape)  # Should output: torch.Size([32, 21])
print("y_train_tensor:", y_train_tensor.shape)  # Should output: torch.Size([32])
print("X_test_tensor:", X_test_tensor.shape)    # Should output: torch.Size([8, 21])
print("y_test_tensor:", y_test_tensor.shape)    # Should output: torch.Size([8])

# Create data loaders for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


# In[ ]:


print(X_train_tensor.shape)  # Should output: torch.Size([32, 18])
print(y_train_tensor.shape)  # Should output: torch.Size([32])


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

# Define Teacher Models with different architectures
class TeacherModel1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

class TeacherModel2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class TeacherModel3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class TeacherModel4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TeacherModel4, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define Student Model
class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)



# In[ ]:


# Initialize the models
input_dim = X_train.shape[1]  # Number of features after preprocessing
output_dim = len(np.unique(y_train))  # Number of unique classes (for classification) or 1 for regression

teacher_model1 = TeacherModel1(input_dim, output_dim)
teacher_model2 = TeacherModel2(input_dim, output_dim)
teacher_model3 = TeacherModel3(input_dim, output_dim)
teacher_model4 = TeacherModel4(input_dim, output_dim)
student_model = StudentModel(input_dim, output_dim)

# Define optimizers and loss function
optimizer_teacher1 = optim.AdamW(teacher_model1.parameters(), lr=0.001, weight_decay=0.01)
optimizer_teacher2 = optim.AdamW(teacher_model2.parameters(), lr=0.001, weight_decay=0.01)
optimizer_teacher3 = optim.AdamW(teacher_model3.parameters(), lr=0.001, weight_decay=0.01)
optimizer_teacher4 = optim.AdamW(teacher_model4.parameters(), lr=0.001, weight_decay=0.01)
optimizer_student = optim.Adam(student_model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()  # For classification, use MSELoss for regression

# Step 1: Train the teacher models
def train_model(model, optimizer, X_train_tensor, y_train_tensor, num_epochs=500):  # Increased epochs
    model.train()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    best_loss = float('inf')
    best_accuracy = 0
    best_model_state = None
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_train_tensor).sum().item() / len(y_train_tensor) * 100
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Early stopping with best model saving
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch} with best accuracy: {best_accuracy:.2f}%")
            model.load_state_dict(best_model_state)
            break
            
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    
    return model


# In[ ]:


import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Example: Define your train_loader if not already defined
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

def train_student(student_model, optimizer_student, train_loader, teacher_models,
                 temperature=3.0, alpha=0.5, num_epochs=10, patience=5):
    
    ce_loss_fn = nn.CrossEntropyLoss()
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for teacher_model in teacher_models:
        teacher_model.eval()

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer_student.zero_grad()
            
            # Student's forward pass
            student_logits = student_model(inputs)
            ce_loss = ce_loss_fn(student_logits, labels)
            
            # Get ensemble predictions from teachers
            teacher_logits_avg = torch.zeros_like(student_logits)
            for teacher_model in teacher_models:
                with torch.no_grad():
                    teacher_logits_avg += teacher_model(inputs)
            teacher_logits_avg /= len(teacher_models)
            
            # Knowledge distillation
            student_soft = F.log_softmax(student_logits / temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits_avg / temperature, dim=1)
            kd_loss = kd_loss_fn(student_soft, teacher_soft) * (temperature ** 2)
            
            # Combined loss
            loss = alpha * ce_loss + (1 - alpha) * kd_loss
            loss.backward()
            optimizer_student.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = student_model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                student_model.load_state_dict(best_model_state)
                break
    
    return student_model

# Example usage:
# Assuming you have defined student_model and teacher_model1,...,teacher_model4,
# and created optimizer_student and train_loader:
teacher_models = [teacher_model1, teacher_model2, teacher_model3, teacher_model4]
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)

# Train the student model using the adjusted training loop
student_model = train_student(student_model, optimizer_student, train_loader,
                              teacher_models, temperature=3.0, alpha=0.5, num_epochs=10)

# After training the student model, add this code to save it
# This should be added after the training loop

def save_model(model, path='student_model.pth'):
    try:
        torch.save(model.state_dict(), path)
        print(f"Model successfully saved to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

# Initialize models and training parameters
input_dim = X_train_tensor.shape[1]
output_dim = len(np.unique(y_train))

# Initialize the models
teacher_models = [
    TeacherModel1(input_dim, output_dim),
    TeacherModel2(input_dim, output_dim),
    TeacherModel3(input_dim, output_dim),
    TeacherModel4(input_dim, output_dim)
]
student_model = StudentModel(input_dim, output_dim)

# Train the student model
student_model = train_student(
    student_model=student_model,
    optimizer_student=optimizer_student,
    train_loader=train_loader,
    teacher_models=teacher_models,
    temperature=3.0,
    alpha=0.3,
    num_epochs=100,
    patience=7
)

# Save the trained student model
save_model(student_model)

# Optional: Save model architecture information
model_info = {
    'input_dim': input_dim,
    'output_dim': output_dim
}
with open('model_info.json', 'w') as f:
    json.dump(model_info, f)


# In[ ]:


from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

# Example: Define your train_loader if not already defined
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



# Convert your training data to tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use torch.float32 for regression

# Train teacher models
teacher_model1 = train_model(teacher_model1, optimizer_teacher1, X_train_tensor, y_train_tensor)
teacher_model2 = train_model(teacher_model2, optimizer_teacher2, X_train_tensor, y_train_tensor)
teacher_model3 = train_model(teacher_model3, optimizer_teacher3, X_train_tensor, y_train_tensor)
teacher_model4 = train_model(teacher_model4, optimizer_teacher4, X_train_tensor, y_train_tensor)

# Step 2: Get predictions from teacher models
teacher_model1.eval()
teacher_model2.eval()
teacher_model3.eval()
teacher_model4.eval()

num_epochs=10
with torch.no_grad():
    teacher_preds1 = teacher_model1(X_train_tensor).cpu().numpy()
    teacher_preds2 = teacher_model2(X_train_tensor).cpu().numpy()
    teacher_preds3 = teacher_model3(X_train_tensor).cpu().numpy()
    teacher_preds4 = teacher_model4(X_train_tensor).cpu().numpy()

# Combine predictions from all teacher models (soft labels)
teacher_preds_combined = (teacher_preds1 + teacher_preds2 + teacher_preds3 + teacher_preds4) / 4  # Average the predictions

# Step 3: Train the student model using teacher predictions as soft labels
student_model.train()

# Convert test features (X_test) to a PyTorch tensor
# Assuming X_test is a pandas DataFrame
X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
y_test_tensor = y_test_tensor.long()

optimizer = optim.Adam(student_model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust learning rate every 10 epochs

# Add this function before the training loop
def augment_data(X, y, noise_factor=0.05):
    X_augmented = X + torch.randn_like(X) * noise_factor
    return X_augmented, y

for epoch in range(num_epochs):
    X_aug, y_aug = augment_data(X_train_tensor, y_train_tensor)
    optimizer.zero_grad()
    outputs = student_model(X_aug)
    loss = criterion(outputs, y_aug)
    loss.backward()
    optimizer.step()

    scheduler.step()  # Adjust learning rate after each epoch

    if epoch % 10 == 0:
        student_model.eval()
        with torch.no_grad():
            # Calculate test accuracy
            student_preds_test = student_model(X_test_tensor)
            _, test_predicted = torch.max(student_preds_test, 1)
            test_accuracy = np.mean(test_predicted.numpy() == y_test_tensor.numpy()) * 100
            print(f'Epoch [{epoch}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%')

            # Optionally, calculate test loss
            test_loss = criterion(student_preds_test, y_test_tensor)
            print(f'Epoch [{epoch}/{num_epochs}], Test Loss: {test_loss.item():.4f}')

            # Calculate training accuracy for monitoring
            student_preds_train = student_model(X_train_tensor)
            _, train_predicted = torch.max(student_preds_train, 1)
            train_accuracy = np.mean(train_predicted.numpy() == y_train_tensor.numpy()) * 100
            print(f'Epoch [{epoch}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%')
            
            # Overfitting check
            if train_accuracy - test_accuracy > 30:
                print("Warning: The model may be overfitting!")
            else:
                print("The model is generalizing well.")


student_model.eval()
with torch.no_grad():
    student_preds = student_model(X_test_tensor).cpu().numpy()


print("Student Model Evaluation:")
if output_dim == 1:  # Regression task
    mse = mean_squared_error(y_test, student_preds)
    print(f'Mean Squared Error: {mse}')
else:  # Classification task
    accuracy = np.mean(np.argmax(student_preds, axis=1) == y_test)
    print(f'Accuracy: {accuracy*100:.2f}%')

# After encoding the target variable (y), add this:
n_classes = len(np.unique(y))
print(f"Number of unique classes (output_dim): {n_classes}")

# Save the number of classes for deployment
class_info = {
    'n_classes': int(n_classes),  # Convert to int for JSON serialization
    'class_names': [str(c) for c in crop_encoder.classes_]  # Save class names
}
with open('class_info.json', 'w') as f:
    json.dump(class_info, f, indent=4)

SERVICE_ACCOUNT_FILE = 'crop-yield-prediction-452008-05e1862c508c.json' 
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = '143Gaf6z2s2hSIWjMZXw6snPuUPSZ6WViYZKmagmCzho'

creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

# Update the write_to_sheet function
def write_to_sheet(results, sheet_name='Sheet3'):
    # Convert all values to strings to ensure compatibility
    if isinstance(results[0], list):
        values = [[str(cell) for cell in row] for row in results]
    else:
        values = [[str(cell) for cell in results]]
    
    body = {'values': values}
    try:
        result = sheet.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=f'{sheet_name}!A1',  # This specifies where to start looking for data
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',  # This makes it append
            body=body
        ).execute()
        print(f"Successfully appended {result.get('updates').get('updatedRows')} rows")
    except Exception as e:
        print(f"Error writing to sheet: {e}")

# First define the function (move this up before we use it)
def get_training_metrics(teacher_models, student_model, train_loader):
    metrics = {
        'MODEL NAME': 'Knowledge Distillation Model',
        'Teacher Training Rate': 0,
        'Teacher Learning Rate': 0.0001,
        'Teacher Training Accuracy': 0,
        'Student Training Rate': 0,
        'Student Learning Rate': 0.001,
        'Student Training Accuracy': 0
    }
    
    # Time and evaluate teacher training
    teacher_start = time.time()
    teacher_accuracies = []
    
    for teacher_model in teacher_models:
        teacher_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = teacher_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = (correct/total) * 100
        teacher_accuracies.append(accuracy)
        print(f"Teacher Model Accuracy: {accuracy:.2f}%")
    
    metrics['Teacher Training Rate'] = f"{time.time() - teacher_start:.2f}"
    metrics['Teacher Training Accuracy'] = f"{sum(teacher_accuracies) / len(teacher_accuracies):.2f}%"
    
    # Time and evaluate student training
    student_start = time.time()
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = student_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    student_accuracy = (correct/total) * 100
    print(f"Student Model Accuracy: {student_accuracy:.2f}%")
    
    metrics['Student Training Rate'] = f"{time.time() - student_start:.2f}"
    metrics['Student Training Accuracy'] = f"{student_accuracy:.2f}%"
    
    return metrics

# Then use it later in the code
metrics = get_training_metrics(teacher_models, student_model, train_loader)
write_to_sheet([list(metrics.values())])  # Just pass the values, sheet name is defaulted to 'Sheet3'

# Then update how we call it
try:
    # Get the metrics
    metrics = get_training_metrics(teacher_models, student_model, train_loader)
    
    # Print metrics for verification
    print("\nMetrics to be sent to Google Sheets:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Write to Google Sheets
    values = list(metrics.values())
    
    print("\nWriting to Google Sheets...")
    write_to_sheet([values])  # Just pass the values, sheet name is defaulted to 'Sheet3'
    print("Values written successfully")

except Exception as e:
    print(f"Error writing metrics to Google Sheets: {e}")  

# Add data augmentation for training
def augment_data(X, y, noise_factor=0.05):
    X_augmented = X + torch.randn_like(X) * noise_factor
    return X_augmented, y  
#!/bin/bash

# Create a temporary deployment directory
mkdir -p deploy_temp

# Copy only the necessary files
cp main.py deploy_temp/
cp model2-ed.py deploy_temp/model2_ed.py
cp "GygaUganda - Station.csv" deploy_temp/
cp student_model.pth deploy_temp/
cp model_info.json deploy_temp/
cp class_info.json deploy_temp/
cp requirements.txt deploy_temp/
cp crop-yield-prediction-452008-05e1862c508c.json deploy_temp/

# Move to deployment directory
cd deploy_temp

# Deploy the function
gcloud functions deploy predict-handler-v2 \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point predict_handler \
  --memory 1024MB \
  --timeout 300s

# Clean up
cd ..
rm -rf deploy_temp 
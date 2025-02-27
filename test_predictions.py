import json
from deploy_model import predict_crop, model, numerical_cols, categorical_cols, scaler

def test_predictions():
    # Print expected columns for debugging
    print("\nExpected numerical columns:", numerical_cols)
    print("Expected categorical columns:", categorical_cols)
    
    # Load test cases
    with open('test_input.json', 'r') as f:
        test_data = json.load(f)
    
    print("\nAvailable columns in test data:", list(test_data['test_cases'][0].keys()))
    
    print("\nTesting predictions:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_data['test_cases'], 1):
        # Extract values in the correct order
        input_data = []
        
        # Add numerical features
        for col in numerical_cols:
            input_data.append(test_case[col])
        
        # Add categorical features
        for col in categorical_cols:
            input_data.append(test_case[col])
        
        # Make prediction
        prediction = predict_crop(input_data, model, numerical_cols, categorical_cols, scaler)
        
        print(f"\nTest Case {i}:")
        print(f"Station: {test_case['STATIONNAME']}")
        print(f"Location: ({test_case['LONGITUDE']}, {test_case['LATITUDE']})")
        print(f"Predicted Crop: {prediction}")
        print("-" * 50)

if __name__ == "__main__":
    test_predictions()
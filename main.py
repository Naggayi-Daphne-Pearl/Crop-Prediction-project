import json
from deploy_model import predict_crop, model, numerical_cols, categorical_cols, scaler, encoder_dict

def predict_handler(request):
    request_json = request.get_json()
    if request_json and 'data' in request_json:
        data = request_json['data']
        prediction = predict_crop(data, model, numerical_cols, categorical_cols, scaler, encoder_dict)
        return json.dumps({'prediction': prediction})
    else:
        return 'Error: No data provided', 400
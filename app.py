# app.py

from flask import Flask, request, jsonify
from functions import preprocess_and_predict
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

def validate_flight_details(flight_details):
    """
    Validate flight details to ensure no field is empty.
    """
    required_fields = ['MONTH', 'DAY_OF_MONTH', 'OP_CARRIER_FL_NUM', 'DEP_HOUR', 'ORIGIN', 'DEST']
    for field in required_fields:
        if field not in flight_details or not flight_details[field]:
            return False
    return True

@app.route('/predict_delay', methods=['POST'])
def predict_delay():
    flight_details = request.json  # Assuming you're sending JSON data
    
    # Validate flight details
    if not validate_flight_details(flight_details):
        return jsonify({'error': 'One or more required fields are missing or empty.'}), 400
    
    # Proceed with prediction
    prediction_results = preprocess_and_predict(flight_details)
    print(prediction_results)
    return jsonify(prediction_results)

if __name__ == '__main__':
    app.run(debug=True)


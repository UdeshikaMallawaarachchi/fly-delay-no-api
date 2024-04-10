import pandas as pd
import numpy as np
import joblib
import requests


def get_noaa_weather_data(station_id):
    url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    response=requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # Extracting some data as an example, add more fields as needed
        weather_data = {
            'temperature': data['properties']['temperature']['value'],
            'windSpeed': data['properties']['windSpeed']['value'],
            'windDirection': data['properties']['windDirection']['value'],
            'visibility':data['properties']['visibility']['value'],
            'precipitation':data['properties']['precipitationLastHour']['value']
        }
        return weather_data
    else:
        print("Failed to fetch data")
        return {}



# Path: preprocess_and_predict.py
# Load the linear model and logistic models
lm = joblib.load('linearmodel.pkl')
logmodels = [joblib.load(f'{i}_logmodel.pkl') for i in range(10)]
model_columns = joblib.load('model_columns.pkl')  # Load the expected column names

# Remove 'ARR_DELAY' and 'DELAY_YN' from model_columns if present
model_columns = [col for col in model_columns if col not in ('ARR_DELAY', 'DELAY_YN')]

def preprocess_and_predict(flight_details):
    """
    Preprocess flight details and predict flight delay.
    """
    # Add 'K' prefix to the ORIGIN airport code if it's not already there and if it's for the contiguous United States
    origin_code = flight_details['ORIGIN']
    station_id = 'K' + origin_code if not origin_code.startswith('K') else origin_code
    print(station_id)
    
    weather_data = get_noaa_weather_data(station_id)
    print(weather_data)
    
    # Check if weather data was successfully fetched
    if not weather_data:
        print("Weather data fetch failed. Using default values.")
        # Use some default or previously known average values
        weather_data = {
            'visibility':1600,
            'temperature': 15,  # Default or previously known avg values
            'windSpeed': 5,
            'precipitationLastHour':0.01
        }
    print("Weather data fetched")
   
        
    # Convert the temperature from Celsius to the expected unit if necessary and map the weather data correctly
    real_time_weather = {
        'DEP_HourlyVisibility': weather_data['visibility'] if weather_data['visibility'] is not None else 0,  
        'DEP_HourlyDryBulbTemperature': weather_data['temperature'] if weather_data['temperature'] is not None else 0,
        'DEP_HourlyWindSpeed': weather_data['windSpeed'] if  weather_data['windSpeed'] is not None else 0,
        'DEP_HourlyPrecipitation': weather_data['precipitation'] if weather_data['precipitation'] is not None else 0 # Example fixed value; adjust based on actual data if available
    }


    # Combine flight details with dummy weather data
    combined_data = {**flight_details, **real_time_weather}

    # Create a DataFrame with a single row of the combined data
    df_input = pd.DataFrame([combined_data])

    # Ensure df_input has the same column order as expected by the model
    for col in model_columns:
        if col not in df_input.columns:
           
            df_input[col] = 0  # Add missing columns as zeros

    df_input = df_input[model_columns]  # Reorder columns to match model's expectation
    

    # Predict delay using the linear model
    predicted_log_delay = lm.predict(df_input)
    print(predicted_log_delay)
    predicted_delay = np.exp(predicted_log_delay)[0]  # Assuming the model predicts log delay
  
    

    # Predict delay occurrence using logistic regression models and average their predictions
    delay_occurrence_probs = np.mean(
        [model.predict_proba(df_input)[:, 1] for model in logmodels], axis=0
    )[0]
    print(delay_occurrence_probs)
    delay_occurrence = 1 if delay_occurrence_probs > 0.46 else 0

    return {
        # 'predicted_delay_in_minutes': max(0, predicted_delay) if delay_occurrence else 0,  # Predicted delay in minutes
        'probability_of_delay': delay_occurrence_probs
    }

